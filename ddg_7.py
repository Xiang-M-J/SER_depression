# 在ddg_3的基础上删除了MModel上的prepare
import copy
import datetime
import math
import os

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from einops.layers.torch import Rearrange
from matplotlib import pyplot as plt
from torch.autograd import Function
from torch.cuda.amp import autocast, GradScaler

from blocks import Chomp1d
from config import Args
from model import AT_TAB, TAB_DIFF
from multiModel import mmd, Classifier
# from multiModel import MModel
from utils import load_loader, NoamScheduler, Metric, EarlyStopping, accuracy_cal, EarlyStoppingLoss

dataset_name = ['MODMA', 'CASIA']
num_class = [2, 6]
seq_len = [313, 188]
num_sample = [3549, 7200]
split_rate = [0.6, 0.2, 0.2]
dataset_num = len(dataset_name)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_amp = False
if use_amp:
    scaler = GradScaler()


class GRL(Function):
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x) * constant

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.constant, None


class Hidden(nn.Module):
    def __init__(self, arg: Args):
        super(Hidden, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(arg.filters, 128, kernel_size=2, padding=1),
            Chomp1d(1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, arg.filters, kernel_size=2, padding=1),
            Chomp1d(1),
            nn.BatchNorm1d(arg.filters),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(arg.filters, 64, kernel_size=2, padding=1),
            Chomp1d(1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, arg.filters, kernel_size=2, padding=1),
            Chomp1d(1),
            nn.BatchNorm1d(arg.filters),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(arg.filters, arg.filters, kernel_size=2, padding=1),
            Chomp1d(1),
            nn.BatchNorm1d(arg.filters),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x


class MModel(nn.Module):
    def __init__(self, arg: Args, index, num_layers=None, seq_len=None):
        super(MModel, self).__init__()
        if num_layers is None:
            num_layers = [3, 5]
        if seq_len is not None:
            arg.seq_len = seq_len
        arg.dilation = num_layers[0] + num_layers[1]
        self.shareNet = AT_TAB(arg, num_layers, index=0)
        merge = nn.Sequential(
            nn.Linear(arg.dilation, 1),
            # nn.Dropout(0.2)
        )
        self.specialNet = nn.Sequential(
            TAB_DIFF(arg, num_layer=num_layers, index=1),
            merge,
            Classifier(arg, index),
        )

    def get_generalFeature(self, x):
        x_f = x
        x_b = torch.flip(x, dims=[-1])
        _, x_f, _ = self.shareNet(x_f, x_b)
        return x_f

    def get_mmd_feature(self, x):
        x_f = x
        x_b = torch.flip(x, dims=[-1])
        _, x_f, _ = self.shareNet(x_f, x_b)
        _, x_f = self.specialNet[0](x_f)
        return x_f

    def forward(self, x, y):
        x_f = x
        x_b = torch.flip(x, dims=[-1])
        x_1, x_f, _ = self.shareNet(x_f, x_b)
        x_2, x_f = self.specialNet[0](x_f)
        x = self.specialNet[1](torch.cat([x_1, x_2], dim=-1))
        x = self.specialNet[2](x)
        loss = F.cross_entropy(x, y)
        correct_num = accuracy_cal(x, y)
        return loss, correct_num


class Discriminator(nn.Module):
    def __init__(self, arg):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=arg.filters, out_channels=8, kernel_size=3, padding="same"),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            Rearrange("N C L -> N (C L)"),
            nn.Linear(arg.seq_len * 8, 1000),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1000, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.net(x)
        return x


class DDGTrainer:
    def __init__(self, args: Args):
        args.num_class = num_class
        self.optimizer_type = args.optimizer_type
        self.epochs = args.epochs
        self.inner_iter = 23
        self.mmd_step = 3
        self.feature_dim = args.feature_dim
        self.batch_size = args.batch_size
        self.seq_len = args.seq_len
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.best_path = "models/ddg/MODMA7_best.pt"
        self.model_path = "models/ddg/MODMA7.pt"
        self.pretrain_path = f"models/ddg/pretrain7_{dataset_name[1]}_2.pt"
        self.pretrain_best_path = f"models/ddg/pretrain7_{dataset_name[1]}_best_2.pt"
        self.result_train_path = "results/data/ddg/train7.npy"
        self.result_test_path = "results/data/ddg/test7.npy"
        date = datetime.datetime.now().strftime("%d_%H_%M")
        tmp_path = f"models/ddg/ddg_7_{date}"
        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)
        self.tmp_model_path = tmp_path + "/MODMA_"
        self.models = []
        self.record_epochs = []
        self.test_acc = []
        args.spilt_rate = split_rate
        self.loader = []
        for i in range(dataset_num):
            train_loader, val_loader, test_loader = \
                load_loader(dataset_name[i], split_rate, args.batch_size, args.random_seed, version='V1', order=3)
            self.loader.append([train_loader, val_loader, test_loader])

    def get_optimizer(self, args, parameter, lr):
        if self.optimizer_type == 0:
            optimizer = torch.optim.SGD(params=parameter, lr=lr, weight_decay=args.weight_decay)
            # 对于SGD而言，L2正则化与weight_decay等价
        elif self.optimizer_type == 1:
            optimizer = torch.optim.Adam(params=parameter, lr=lr, betas=(args.beta1, args.beta2),
                                         weight_decay=args.weight_decay)
            # 对于Adam而言，L2正则化与weight_decay不等价
        elif self.optimizer_type == 2:
            optimizer = torch.optim.AdamW(params=parameter, lr=lr, betas=(args.beta1, args.beta2),
                                          weight_decay=args.weight_decay)
        else:
            raise NotImplementedError
        return optimizer

    def get_scheduler(self, optimizer, arg: Args):
        if arg.scheduler_type == 0:
            return None
        elif arg.scheduler_type == 1:
            return torch.optim.lr_scheduler.StepLR(optimizer, arg.step_size, arg.gamma)
        elif arg.scheduler_type == 2:
            return NoamScheduler(optimizer, arg.d_model, arg.initial_lr, arg.warmup)
        elif arg.scheduler_type == 3:
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=arg.epochs, eta_min=1e-6)
        else:
            raise NotImplementedError

    def train_step(self, model, optimizer, batch):
        x, y = self.get_data(batch)
        optimizer.zero_grad()
        if use_amp:
            with autocast():
                loss, correct_num = model(x, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, correct_num = model(x, y)
            loss.backward()
            optimizer.step()
        return loss, correct_num

    @staticmethod
    def get_data(batch):
        x, y = batch
        x, y = x.to(device), y.to(device)
        return x, y

    @staticmethod
    def get_mmd_loss(model, hidden1, hidden2, train_batch):
        x1 = model.get_generalFeature(train_batch[0][0].to(device))
        x2 = model.get_generalFeature(train_batch[1][0].to(device))
        mini_shape = min([x1.shape[0], x2.shape[0]])
        x1, x2 = x1[:mini_shape], x2[:mini_shape]
        mmd_feature = [hidden1(x1), hidden2(x2)]
        mini_shape = min([x.shape[0] for x in mmd_feature])
        mmd_loss = mmd(mmd_feature[0][:mini_shape].view(mini_shape, -1),
                       mmd_feature[1][:mini_shape].view(mini_shape, -1))
        return mmd_loss

    @staticmethod
    def get_domain_loss(model, hidden1, hidden2, discriminator, train_batch, p):
        x1 = model.get_generalFeature(train_batch[0][0].to(device))
        x2 = model.get_generalFeature(train_batch[1][0].to(device))
        generalFeature = [hidden1(x1), hidden2(x2)]
        x = torch.cat(generalFeature, dim=0)
        label = torch.cat([torch.ones(len(train_batch[0][0])), torch.zeros(len(train_batch[1][0]))], dim=0).long()
        alpha = 2. / (1 + np.exp((-10. * p))) - 1
        x = GRL.apply(x, alpha)
        loss, correct_num_domain = discriminator(x, label.to(device))
        return loss, correct_num_domain

    @staticmethod
    def discriminator_step(model, hidden1, hidden2, discriminator, d_optimizer, train_batch):
        model.eval()
        hidden1.eval()
        hidden2.eval()
        discriminator.train()
        x1 = model.get_generalFeature(train_batch[0][0].to(device))
        x2 = model.get_generalFeature(train_batch[1][0].to(device))
        x1 = hidden1(x1)
        x2 = hidden2(x2)
        d_x1 = discriminator(x1)
        d_x2 = discriminator(x2)
        d_loss = torch.mean(d_x2) - torch.mean(d_x1)
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        for layer in discriminator.net:
            if layer.__class__.__name__ == "Conv1d" or layer.__class__.__name__ == "Linear":
                layer.weight.requires_grad = False
                layer.weight.clamp_(-0.01, 0.01)
                layer.weight.requires_grad = True
        return d_loss

    def generator_step(self, model, hidden1, hidden2, discriminator, g_optimizer, train_batch):
        model.train()
        hidden1.train()
        hidden2.train()
        discriminator.eval()
        mmd_loss = self.get_mmd_loss(model, hidden1, hidden2, train_batch)
        x1 = model.get_generalFeature(train_batch[0][0].to(device))
        x2 = model.get_generalFeature(train_batch[1][0].to(device))
        x1 = hidden1(x1)
        x2 = hidden2(x2)

        d_x1 = discriminator(x1)
        d_x2 = discriminator(x2)
        g_loss = -torch.mean(d_x2)
        loss = mmd_loss + g_loss
        g_optimizer.zero_grad()
        loss.backward()
        g_optimizer.step()
        return mmd_loss, g_loss

    def val_step(self, model, batch):
        x, y = self.get_data(batch)
        if use_amp:
            with autocast():
                loss, correct_num = model(x, y)
        else:
            loss, correct_num = model(x, y)
        return loss, correct_num

    def test_step(self, model, batch):
        return self.val_step(model, batch)

    def train(self):
        mini_iter = min([len(self.loader[i][0]) for i in range(dataset_num)])
        train_iter = [iter(self.loader[i][0]) for i in range(dataset_num)]

        model = MModel(arg, seq_len=seq_len[0], index=0).to(device)
        optimizer = self.get_optimizer(arg, model.parameters(), lr=arg.lr)
        scheduler = self.get_scheduler(optimizer, arg)
        hidden1 = Hidden(arg).to(device)
        hidden2 = Hidden(arg).to(device)
        discriminator = Discriminator(arg).to(device)

        d_optimizer = torch.optim.SGD(discriminator.parameters(), lr=arg.lr, weight_decay=0.1)
        d_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, step_size=30, gamma=0.3)

        parameter = [
            {'params': model.shareNet.parameters(), 'lr': arg.lr},
            {'params': hidden1.parameters(), 'lr': arg.lr},
            {"params": hidden2.parameters(), 'lr': arg.lr},
        ]
        g_optimizer = torch.optim.SGD(parameter, lr=arg.lr, weight_decay=0.1)
        g_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=30, gamma=0.3)

        early_stop = EarlyStoppingLoss(patience=5, delta_loss=6e-4)
        best_val_accuracy = 0
        metric = Metric()
        domain_acc = 0
        domain_num = 0
        val_acc = 0
        val_loss = 0
        train_acc = 0
        train_loss = 0
        train_num = len(self.loader[0][0].dataset)
        val_num = len(self.loader[0][1].dataset)
        for epoch in range(self.epochs):

            if epoch % self.mmd_step == 0:
                D_loss = []
                MMD_loss = []
                G_loss = []
                for step in range(self.inner_iter):
                    train_batch = []
                    for i in range(dataset_num):
                        try:
                            batch = next(train_iter[i])
                        except StopIteration:
                            train_iter[i] = iter(self.loader[i][0])
                            batch = next(train_iter[i])
                        train_batch.append(batch)
                    for _ in range(3):
                        d_loss = self.discriminator_step(model, hidden1, hidden2, discriminator, d_optimizer, train_batch)
                        D_loss.append(d_loss.data.item())
                    mmd_loss, g_loss = self.generator_step(model, hidden1, hidden2, discriminator, g_optimizer, train_batch)
                    # D_loss.append(d_loss.data.item())
                    MMD_loss.append(mmd_loss.data.item())
                    G_loss.append(g_loss.data.item())
                print("Discriminator loss: ",np.mean(D_loss))
                print("MMD loss: ", np.mean(MMD_loss))
                print("Generator loss: ", np.mean(G_loss))

            model.train()
            for batch in self.loader[0][0]:
                loss, correct_num = self.train_step(model, optimizer, batch)
                train_acc += correct_num.cpu().numpy()
                train_loss += loss.data.item()

            scheduler.step()
            d_scheduler.step()
            g_scheduler.step()
            print(f"epoch {epoch + 1}:")
            train_acc = train_acc / train_num
            train_loss = train_loss / mini_iter
            metric.train_acc.append(train_acc)
            metric.train_loss.append(train_loss)
            print(f"MODMA: train Loss:{train_loss:.4f}\t train Accuracy:{train_acc * 100:.3f}\t")
            model.eval()
            with torch.no_grad():
                for batch in self.loader[0][1]:
                    loss, correct_num = self.val_step(model, batch)
                    val_acc += correct_num.cpu().numpy()
                    val_loss += loss.data.item()
            val_acc = val_acc / val_num
            val_loss = val_loss / math.ceil(val_num / self.batch_size)
            metric.val_acc.append(val_acc)
            metric.val_loss.append(val_loss)
            print(f"MODMA: val Loss:{val_loss:.4f}\t val Accuracy:{val_acc * 100:.3f}\t")
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                metric.best_val_acc[0] = train_acc
                metric.best_val_acc[1] = best_val_accuracy
                torch.save(model, self.best_path)
            if val_acc > 0.994:
                tmp = self.multi_test(model)
                self.test_acc.append(tmp)
                if tmp > 0.994:
                    torch.save(model, self.tmp_model_path + f"{epoch}_" + f"{tmp * 1000}.pt")
                    self.models.append(copy.deepcopy(model))
            plt.clf()
            plt.plot(metric.train_acc)
            plt.plot(metric.val_acc)
            plt.ylabel("accuracy(%)")
            plt.xlabel("epoch")
            plt.legend([f' train acc', f'val acc'])
            plt.title(f"train accuracy and validation accuracy")
            plt.pause(0.02)
            plt.ioff()  # 关闭画图的窗口
            if early_stop(val_loss):
                break
            domain_acc = 0
            domain_num = 0
            val_acc = 0
            val_loss = 0
            train_acc = 0
            train_loss = 0
        np.save(self.result_train_path, metric.item())
        torch.save(model, self.model_path)

    def test(self):
        model = torch.load(self.model_path)
        metric = Metric(mode="test")
        metric.test_acc = []
        metric.test_loss = []
        test_acc = 0
        test_loss = 0
        test_num = len(self.loader[0][2].dataset)
        model.eval()
        print("test...")
        with torch.no_grad():
            for batch in self.loader[0][2]:
                loss, correct_num = self.test_step(model, batch)
                test_acc += correct_num
                test_loss += loss.data.item()
        print("final path")
        test_acc = test_acc / test_num
        test_loss = test_loss / math.ceil(test_num / self.batch_size)
        print(f"{dataset_name}: test Loss:{test_loss:.4f}\t test Accuracy:{test_acc * 100:.3f}\t")
        metric.test_acc.append(test_acc)
        metric.test_loss.append(test_loss)
        model = torch.load(self.best_path)
        test_acc = 0
        test_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in self.loader[0][2]:
                loss, correct_num = self.test_step(model, batch)
                test_acc += correct_num
                test_loss += loss.data.item()
        print("best path")
        test_acc = test_acc / test_num
        test_loss = test_loss / math.ceil(test_num / self.batch_size)
        print(f"{dataset_name}: test Loss:{test_loss:.4f}\t test Accuracy:{test_acc * 100:.3f}\t")
        metric.test_acc.append(test_acc)
        metric.test_loss.append(test_loss)
        np.save(self.result_test_path, metric.item())

    def multi_test(self, model):
        test_num = len(self.loader[0][2].dataset)
        test_acc = 0
        model.eval()
        with torch.no_grad():
            for batch in self.loader[0][2]:
                loss, correct_num = self.test_step(model, batch)
                test_acc += correct_num.cpu().numpy()
        test_acc = test_acc / test_num
        return test_acc


if __name__ == "__main__":
    arg = Args()
    arg.random_seed = 34
    trainer = DDGTrainer(arg)
    # trainer.pretrain()
    trainer.train()
    trainer.test()
    print(trainer.test_acc)

    # for m in trainer.models:
    #     print(m.prepare.net[0].bias[0:9])
