# 将mmd loss 和 disc loss相加后一起反向传播
import math

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from einops.layers.torch import Rearrange
from matplotlib import pyplot as plt
from torch.autograd import Function

from config import Args
from multiModel import MModel, mmd
from utils import load_loader, NoamScheduler, Metric, accuracy_cal

dataset_name = ['MODMA', 'DAIC21']  # target source
num_class = [2, 2]
seq_len = [313, 188]
num_sample = [3549, 24000]
split_rate = [0.6, 0.2, 0.2]
dataset_num = len(dataset_name)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GRL(Function):
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x) * constant

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.constant, None


class Discriminator(nn.Module):
    def __init__(self, arg):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 1),
            Rearrange("N C L -> N (C L)"),
            nn.Linear(arg.seq_len, 2),
        )

    def forward(self, x, y):
        x = self.net(x)
        loss = F.cross_entropy(x, y, label_smoothing=0.1)
        correct_num = accuracy_cal(x, y)
        return loss, correct_num


class DANNModel(nn.Module):
    def __init__(self, arg: Args, ):
        super(DANNModel, self).__init__()
        self.lstm = nn.Sequential(
            Rearrange("N C L -> N L C"),
            nn.LSTM(input_size=arg.feature_dim, hidden_size=256, batch_first=True, bidirectional=False)
        )
        self.classifier = nn.Sequential(
            Rearrange("N L C -> N C L"),
            nn.AdaptiveAvgPool1d(1),
            Rearrange("N C L -> N (C L)"),
            nn.Linear(256, 2)
        )

    def get_Feature(self, x):
        self.lstm[1].flatten_parameters()
        x, (_, _) = self.lstm(x)
        return x

    def forward(self, x, y):
        self.lstm[1].flatten_parameters()  # 调用flatten_parameters让parameter的数据存放成连续的块，提高内存的利用率和效率
        RNN_out, (_, _) = self.lstm(x)  # if bidirectional=True，输出将包含序列中每个时间步的正向和反向隐藏状态的串联
        x = self.classifier(RNN_out)
        loss = F.cross_entropy(x, y)
        correct_num = accuracy_cal(x, y)
        return loss, correct_num


class UDATrainer:
    def __init__(self, args: Args):
        args.num_class = num_class
        self.optimizer_type = args.optimizer_type
        self.epochs = args.epochs
        self.feature_dim = args.feature_dim
        self.batch_size = args.batch_size
        self.seq_len = args.seq_len
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.best_path = "models/uda/MODMA_ddan_best.pt"
        self.model_path = "models/uda/MODMA_ddan.pt"
        self.result_train_path = "results/data/uda/train_ddan.npy"
        self.result_test_path = "results/data/uda/test_ddan.npy"
        self.tmp_model_path = "models/uda/MODMA_ddan_"
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
    def get_mmd_loss(model, src_model: MModel, hidden1, hidden2, train_batch):
        x1 = model.get_generalFeature(train_batch[0][0].to(device))
        x2 = train_batch[1][0].to(device)
        x2_f = src_model.prepare(x2)
        x2_b = src_model.prepare(torch.flip(x2, dims=[-1]))
        _, x2, _ = model.shareNet(x2_f, x2_b)
        mini_shape = min([x1.shape[0], x2.shape[0]])
        x1, x2 = x1[:mini_shape], x2[:mini_shape]
        mmd_feature = [hidden1(x1), hidden2(x2)]
        mini_shape = min([x.shape[0] for x in mmd_feature])
        mmd_loss = mmd(mmd_feature[0][:mini_shape].view(mini_shape, -1),
                       mmd_feature[1][:mini_shape].view(mini_shape, -1))
        return mmd_loss

    def DANN_step(self, model, discriminator, train_batch, p):
        tgt_x, tgt_y = self.get_data(train_batch[0])
        src_x, src_y = self.get_data(train_batch[1])
        src_label_loss, src_correct_num = model(src_x, src_y)
        tgt_label_loss, _ = model(tgt_x, tgt_y)
        x1 = model.get_Feature(tgt_x)
        x2 = model.get_Feature(src_x)
        x = torch.cat([x1, x2], dim=0)
        label = torch.cat([torch.ones(len(train_batch[0][0])), torch.zeros(len(train_batch[1][0]))], dim=0).long()
        alpha = 2. / (1 + np.exp((-10. * p))) - 1
        x = GRL.apply(x, alpha)
        domain_loss, correct_num_domain = discriminator(x, label.to(device))
        loss = domain_loss + src_label_loss
        return loss, correct_num_domain, src_correct_num

    def val_step(self, model, batch):
        x, y = self.get_data(batch)
        loss, correct_num = model(x, y)
        return loss, correct_num

    def test_step(self, model, batch):
        return self.val_step(model, batch)

    def train(self):
        mini_iter = min([len(self.loader[i][0]) for i in range(dataset_num)])
        train_iter = [iter(self.loader[i][0]) for i in range(dataset_num)]
        model = DANNModel(arg).to(device)
        discriminator = Discriminator(arg).to(device)
        discriminator.train()
        parameter = [
            {'params': model.parameters(), 'lr': arg.lr},
            {"params": discriminator.parameters(), 'lr': arg.lr}
        ]
        optimizer = torch.optim.AdamW(parameter, lr=arg.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.3)
        best_val_accuracy = 0
        metric = Metric()
        domain_acc = 0
        domain_num = 0
        val_acc = 0
        val_loss = 0
        train_acc = 0
        train_num = 0
        train_loss = 0
        val_num = len(self.loader[0][1].dataset)
        for epoch in range(self.epochs):
            model.train()
            for step in range(mini_iter):
                train_batch = []
                for i in range(dataset_num):
                    try:
                        batch = next(train_iter[i])
                    except StopIteration:
                        train_iter[i] = iter(self.loader[i][0])
                        batch = next(train_iter[i])
                    train_batch.append(batch)
                p = epoch / self.epochs
                loss, correct_num, src_correct_num = self.DANN_step(model, discriminator, train_batch, p)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                domain_acc += correct_num.cpu().numpy()
                train_acc += src_correct_num.cpu().numpy()
                train_loss += loss.data.item()
                train_num += len(train_batch[1][0])
                domain_num += len(train_batch[1][0])
                domain_num += len(train_batch[0][0])
            domain_acc = domain_acc / domain_num
            print(f"domain accuracy: {domain_acc:.3f}")
            scheduler.step()
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
            # if val_acc > 0.994:
            #     tmp = self.multi_test(model)
            #     self.test_acc.append(tmp)
            #     if tmp > 0.994:
            #         torch.save(model, self.tmp_model_path + f"{epoch}_" + f"{tmp * 1000}.pt")
            #         self.models.append(copy.deepcopy(model))
            plt.clf()
            plt.plot(metric.train_acc)
            plt.plot(metric.val_acc)
            plt.ylabel("accuracy(%)")
            plt.xlabel("epoch")
            plt.legend([f' train acc', f'val acc'])
            plt.title(f"train accuracy and validation accuracy")
            plt.pause(0.02)
            plt.ioff()  # 关闭画图的窗口

            domain_acc = 0
            domain_num = 0
            val_acc = 0
            val_loss = 0
            train_acc = 0
            train_num = 0
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
    trainer = UDATrainer(arg)
    trainer.train()
    trainer.test()
    print(trainer.test_acc)
    # for m in trainer.models:
    #     print(m.prepare.net[0].bias[0:9])
