import math
import os

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from einops.layers.torch import Rearrange
from matplotlib import pyplot as plt
from torch.autograd import Function
from torch.cuda.amp import GradScaler

from config import Args
from model import AT_TAB, TAB_DIFF
from utils import get_newest_file, load_loader, NoamScheduler, Metric, accuracy_cal

# 联合训练
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


class BaseModel(nn.Module):
    def __init__(self, arg: Args, num_layers=None):
        super(BaseModel, self).__init__()
        if num_layers is None:
            num_layers = [3, 5]
        self.prepare1 = nn.Sequential(
            nn.Conv1d(in_channels=arg.feature_dim, out_channels=arg.filters, kernel_size=1, dilation=1, padding=0),
            nn.BatchNorm1d(arg.filters),
            nn.ReLU(),
        )
        self.prepare2 = nn.Sequential(
            nn.Conv1d(in_channels=arg.feature_dim, out_channels=arg.filters, kernel_size=1, dilation=1, padding=0),
            nn.BatchNorm1d(arg.filters),
            nn.ReLU(),
        )
        self.generalExtractor = AT_TAB(arg, num_layer=num_layers, index=0)
        self.specialExtractor1 = TAB_DIFF(arg, num_layer=num_layers, index=1)
        self.specialExtractor2 = TAB_DIFF(arg, num_layer=num_layers, index=1)
        arg.dilation = num_layers[0] + num_layers[1]

        self.classifier1 = nn.Sequential(
            nn.Linear(arg.dilation, 1),
            Rearrange("N C L H -> N C (L H)"),
            nn.AdaptiveAvgPool1d(1),
            Rearrange('N C L -> N (C L)'),
            nn.Linear(in_features=arg.filters, out_features=arg.num_class[0])
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(arg.dilation, 1),
            Rearrange("N C L H -> N C (L H)"),
            nn.AdaptiveAvgPool1d(1),
            Rearrange('N C L -> N (C L)'),
            nn.Linear(in_features=arg.filters, out_features=arg.num_class[1])
        )

    def forward(self, x, y, index=0, mask=None):
        if index == 0:  # MODMA
            x_f = self.prepare1(x)
            x_b = self.prepare1(torch.flip(x, dims=[-1]))
            x_1, x, _ = self.generalExtractor(x_f, x_b, mask)
            x_2, x = self.specialExtractor1(x)
            x = self.classifier1(torch.cat([x_1, x_2], dim=-1))
        else:  # CASIA
            x_f = self.prepare2(x)
            x_b = self.prepare2(torch.flip(x, dims=[-1]))
            x_1, x, _ = self.generalExtractor(x_f, x_b, mask)
            x_2, x = self.specialExtractor2(x)
            x = self.classifier2(torch.cat([x_1, x_2], dim=-1))
        loss = F.cross_entropy(x, y)
        correct_num = accuracy_cal(x, y)
        return loss, correct_num


class Trainer:
    def __init__(self, args: Args):
        args.num_class = num_class
        self.optimizer_type = args.optimizer_type
        self.epochs = args.epochs
        self.iteration = 5000
        self.inner_iter = 20
        self.mmd_step = 6
        self.test_acc = []
        self.feature_dim = args.feature_dim
        self.batch_size = args.batch_size
        self.seq_len = args.seq_len
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.best_path = "models/new/MODMA_best.pt"
        self.model_path = "models/new/MODMA.pt"
        self.result_train_path = "results/data/new/train.npy"
        self.result_test_path = "results/data/new/test.npy"
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

    def train_step(self, model, optimizer, batch, index):
        x, y = self.get_data(batch)
        loss, correct_num = model(x, y, index)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss, correct_num

    def train_step_(self, model, batch, index):
        x, y = self.get_data(batch)
        loss, correct_num = model(x, y, index)
        return loss, correct_num

    @staticmethod
    def get_data(batch):
        x, y = batch
        x, y = x.to(device), y.to(device)
        return x, y

    @staticmethod
    def get_domain_loss(model, src_model, hidden, discriminator, train_batch, p):
        x1 = model.get_generalFeature(train_batch[0][0].to(device))
        x2 = src_model.get_generalFeature(train_batch[1][0].to(device))
        generalFeature = [hidden(x1), hidden(x2)]
        x = torch.cat(generalFeature, dim=0)
        label = torch.cat([torch.ones(len(train_batch[0][0])), torch.zeros(len(train_batch[1][0]))], dim=0).long()
        alpha = 2. / (1 + np.exp((-10. * p))) - 1
        x = GRL.apply(x, alpha)
        loss, correct_num_domain = discriminator(x, label.to(device))
        return loss, correct_num_domain

    def val_step(self, model, batch, index):
        x, y = self.get_data(batch)
        loss, correct_num = model(x, y, index)
        return loss, correct_num

    def test_step(self, model, batch, index):
        return self.val_step(model, batch, index)

    def train_(self):
        mini_iter = min([len(self.loader[i][0]) for i in range(dataset_num)])
        train_iter = [iter(self.loader[i][0]) for i in range(dataset_num)]
        model = BaseModel(arg).to(device)

        optimizer = self.get_optimizer(arg, model.parameters(), lr=arg.lr)
        scheduler = self.get_scheduler(optimizer, arg)
        best_val_accuracy = 0
        metric = Metric()
        val_acc = 0
        val_loss = 0
        train_acc = 0
        train_num = 0
        train_loss = 0
        for epoch in range(self.epochs):
            model.train()
            train_batch = []
            for step in range(mini_iter):
                for i in range(dataset_num):
                    try:
                        batch = next(train_iter[i])
                    except StopIteration:
                        train_iter[i] = iter(self.loader[i][0])
                        batch = next(train_iter[i])
                    train_batch.append(batch)
                loss1, correct_num = self.train_step_(model, train_batch[0], 0)
                train_acc += correct_num.cpu().numpy()
                train_loss += loss1.data.item()
                loss2, _ = self.train_step_(model, train_batch[1], 1)
                loss = loss1 + loss2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            print(f"epoch {epoch + 1}:")
            train_acc = train_acc / len(self.loader[0][0].dataset)
            train_loss = train_loss / mini_iter
            metric.train_acc.append(train_acc)
            metric.train_loss.append(train_loss)
            print(f"MODMA: train Loss:{train_loss:.4f}\t train Accuracy:{train_acc * 100:.3f}\t")
            model.eval()
            with torch.no_grad():
                for batch in self.loader[0][1]:
                    loss, correct_num = self.val_step(model, batch, 0)
                    val_acc += correct_num.cpu().numpy()
                    val_loss += loss.data.item()
            val_acc = val_acc / int(num_sample[0] * split_rate[1])
            val_loss = val_loss / math.ceil(int(num_sample[0] * split_rate[1]) / self.batch_size)
            metric.val_acc.append(val_acc)
            metric.val_loss.append(val_loss)
            print(f"MODMA: val Loss:{val_loss:.4f}\t val Accuracy:{val_acc * 100:.3f}\t")
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                metric.best_val_acc[0] = train_acc
                metric.best_val_acc[1] = best_val_accuracy
                torch.save(model, self.best_path)

            if val_acc > 0.994:
                self.test_acc.append(self.multi_test(model))

            plt.clf()
            plt.plot(metric.train_acc)
            plt.plot(metric.val_acc)
            plt.ylabel("accuracy(%)")
            plt.xlabel("epoch")
            plt.legend([f' train acc', f'val acc'])
            plt.title(f"train accuracy and validation accuracy")
            plt.pause(0.02)
            plt.ioff()  # 关闭画图的窗口
            val_acc = 0
            val_loss = 0
            train_acc = 0
            train_num = 0
            train_loss = 0
        np.save(self.result_train_path, metric.item())
        torch.save(model, self.model_path)

    def train(self):
        # 训练集快速收敛，但是验证集不行
        mini_iter = min([len(self.loader[i][0]) for i in range(dataset_num)])
        train_iter = [iter(self.loader[i][0]) for i in range(dataset_num)]
        model = BaseModel(arg).to(device)
        parameter1 = [
            {"params": model.prepare1.parameters(), "lr": arg.lr},
            {"params": model.generalExtractor.parameters(), 'lr': arg.lr},
            {"params": model.specialExtractor1.parameters(), 'lr': arg.lr},
            {"params": model.classifier1.parameters(), 'lr': arg.lr}
        ]
        parameter2 = [
            {"params": model.prepare2.parameters(), "lr": 0.5 * arg.lr},
            {"params": model.generalExtractor.parameters(), 'lr': 0.1 * arg.lr},
            {"params": model.specialExtractor2.parameters(), 'lr': 0.5 * arg.lr},
            {"params": model.classifier2.parameters(), 'lr': 0.5 * arg.lr}
        ]
        optimizer1 = self.get_optimizer(arg, parameter1, lr=arg.lr)
        scheduler1 = self.get_scheduler(optimizer1, arg)
        optimizer2 = self.get_optimizer(arg, parameter2, lr=arg.lr)
        scheduler2 = self.get_scheduler(optimizer2, arg)
        best_val_accuracy = 0
        metric = Metric()
        val_acc = 0
        val_loss = 0
        train_acc = 0
        train_num = 0
        train_loss = 0
        for epoch in range(self.epochs):
            model.train()
            for batch in self.loader[0][0]:
                loss, correct_num = self.train_step(model, optimizer1, batch, 0)
                train_acc += correct_num.cpu().numpy()
                train_loss += loss.data.item()

            if (epoch + 1) % 6 == 0:
                for batch in self.loader[1][0]:
                    self.train_step(model, optimizer2, batch, 1)

            scheduler1.step()
            scheduler2.step()
            print(f"epoch {epoch + 1}:")
            train_acc = train_acc / len(self.loader[0][0].dataset)
            train_loss = train_loss / mini_iter
            metric.train_acc.append(train_acc)
            metric.train_loss.append(train_loss)
            print(f"MODMA: train Loss:{train_loss:.4f}\t train Accuracy:{train_acc * 100:.3f}\t")
            model.eval()
            with torch.no_grad():
                for batch in self.loader[0][1]:
                    loss, correct_num = self.val_step(model, batch, 0)
                    val_acc += correct_num.cpu().numpy()
                    val_loss += loss.data.item()
            val_acc = val_acc / int(num_sample[0] * split_rate[1])
            val_loss = val_loss / math.ceil(int(num_sample[0] * split_rate[1]) / self.batch_size)
            metric.val_acc.append(val_acc)
            metric.val_loss.append(val_loss)
            print(f"MODMA: val Loss:{val_loss:.4f}\t val Accuracy:{val_acc * 100:.3f}\t")
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                metric.best_val_acc[0] = train_acc
                metric.best_val_acc[1] = best_val_accuracy
                torch.save(model, self.best_path)

            if val_acc > 0.994:
                self.test_acc.append(self.multi_test(model))

            plt.clf()
            plt.plot(metric.train_acc)
            plt.plot(metric.val_acc)
            plt.ylabel("accuracy(%)")
            plt.xlabel("epoch")
            plt.legend([f' train acc', f'val acc'])
            plt.title(f"train accuracy and validation accuracy")
            plt.pause(0.02)
            plt.ioff()  # 关闭画图的窗口
            val_acc = 0
            val_loss = 0
            train_acc = 0
            train_num = 0
            train_loss = 0
        np.save(self.result_train_path, metric.item())
        torch.save(model, self.model_path)

    def test(self, path=None):

        if path is None:
            path = get_newest_file("models/Multi/")
            print(f"path is None, choose the newest model: {path}")
        if not os.path.exists(path):
            print(f"error! cannot find the {path}")
            return
        model = torch.load(self.model_path)
        metric = Metric(mode="test")
        metric.test_acc = []
        metric.test_loss = []
        test_acc = 0
        test_loss = 0
        test_num = num_sample[0] - int(num_sample[0] * split_rate[0]) - int(num_sample[0] * split_rate[1])
        model.eval()
        print("test...")
        with torch.no_grad():
            for batch in self.loader[0][2]:
                loss, correct_num = self.test_step(model, batch, 0)
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
                loss, correct_num = self.test_step(model, batch, 0)
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
        test_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in self.loader[0][2]:
                loss, correct_num = self.test_step(model, batch, 0)
                test_acc += correct_num.cpu().numpy()
                test_loss += loss.data.item()
        test_acc = test_acc / test_num
        return test_acc


if __name__ == "__main__":
    arg = Args()
    trainer = Trainer(arg)
    trainer.train()
    trainer.test()
    print(trainer.test_acc)