import math
import os

import numpy as np
from matplotlib import pyplot as plt
from torch.autograd import Function
import torch.optim
from torch.cuda.amp import autocast, GradScaler

from config import Args
from multiModel import MModel, Discriminator, mmd
from utils import get_newest_file, load_loader, NoamScheduler, Metric

dataset_name = ['MODMA', 'IEMOCAP']
num_class = [2, 6]
seq_len = [313, 313]
num_sample = [3549, 7766]
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


class MultiTrainer:
    def __init__(self, args: Args):
        args.num_class = num_class
        # args.seq_len = seq_len
        self.optimizer_type = args.optimizer_type
        self.epochs = args.epochs
        self.iteration = 5000
        self.inner_iter = 15
        self.mmd_step = 8
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.best_path = "models/Multi/MODMA_best.pt"
        self.model_path = "models/Multi/MODMA.pt"
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
    def get_mmd_loss(model, train_batch):
        mmd_feature = []
        for i in range(dataset_num):
            x = model.get_generalFeature(train_batch[i][0].to(device))
            mmd_feature.append(x)
        mini_shape = min([x.shape[0] for x in mmd_feature])
        mmd_loss = mmd(mmd_feature[0][:mini_shape].view(mini_shape, -1),
                       mmd_feature[1][:mini_shape].view(mini_shape, -1))
        return mmd_loss

    @staticmethod
    def get_domain_loss(model, discriminator, train_batch, p):
        generalFeature = []
        for i in range(dataset_num):
            x = model[i].get_generalFeature(train_batch[i][0].to(device))
            generalFeature.append(x)
        x = torch.cat(generalFeature, dim=0)
        label = torch.cat([torch.ones(len(train_batch[0][0])), torch.zeros(len(train_batch[1][0]))], dim=0).long()
        alpha = 2. / (1 + np.exp((-10. * p))) - 1
        x = GRL.apply(x, alpha)
        loss, correct_num_domain = discriminator(x, label.to(device))
        return loss, correct_num_domain

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

    @staticmethod
    def get_data(batch):
        x, y = batch
        x, y = x.to(device), y.to(device)
        return x, y

    def train(self):
        # arg.step_size = 50
        # arg.gamma = 0.5
        mini_iter = min([len(self.loader[i][0]) for i in range(dataset_num)])
        # mini_sample_num = min(len(self.src1_train.dataset), len(self.src2_train.dataset), len(self.tgt_train.dataset))

        train_iter = [iter(self.loader[i][0]) for i in range(dataset_num)]
        model = MModel(arg, seq_len=seq_len[0], index=0).to(device)
        optimizer = self.get_optimizer(arg, model.parameters(), lr=arg.lr)
        scheduler = self.get_scheduler(optimizer, arg)
        parameter = [
            {'params': model.prepare.parameters(), 'lr': arg.lr},
            {'params': model.shareNet.parameters(), 'lr': arg.lr}
        ]
        share_optimizer = self.get_optimizer(arg, parameter, lr=arg.lr)
        share_scheduler = self.get_scheduler(share_optimizer, arg)

        best_val_accuracy = 0
        metric = Metric()
        train_num = 0
        domain_acc = 0
        tgt_acc = 0
        domain_num = 0
        val_acc = 0
        val_loss = 0
        train_acc = 0
        train_loss = 0
        for epoch in range(self.epochs):

            model.train()
            if (epoch + 1) % self.mmd_step == 0:
                m_loss = []
                for step in range(self.inner_iter):
                    train_batch = []
                    for i in range(dataset_num):
                        try:
                            batch = next(train_iter[i])
                        except StopIteration:
                            train_iter[i] = iter(self.loader[i][0])
                            batch = next(train_iter[i])
                        train_batch.append(batch)
                    mmd_loss = self.get_mmd_loss(model, train_batch)
                    m_loss.append(mmd_loss.data.item())
                    share_optimizer.zero_grad()
                    mmd_loss.backward()
                    share_optimizer.step()
                print(np.mean(m_loss))

            for batch in self.loader[0][0]:
                loss, correct_num = self.train_step(model, optimizer, batch)
                train_acc += correct_num.cpu().numpy()
                train_loss += loss.data.item()

            scheduler.step()
            share_scheduler.step()
            print(f"epoch {epoch + 1}:")
            train_acc = train_acc / len(self.loader[0][0].dataset)
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

            plt.clf()

            plt.plot(metric.train_acc)
            plt.plot(metric.val_acc)
            plt.ylabel("accuracy(%)")
            plt.xlabel("epoch")
            plt.legend([f' train acc', f'val acc'])
            plt.title(f"train accuracy and validation accuracy")
            plt.pause(0.02)
            plt.ioff()  # 关闭画图的窗口

            train_num = 0
            domain_acc = 0
            tgt_acc = 0
            domain_num = 0
            val_acc = 0
            val_loss = 0
            train_acc = 0
            train_loss = 0
        np.save("results/data/Multi/MODMA.npy", metric.item())
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
        np.save("results/data/Multi/test.npy", metric.item())


if __name__ == "__main__":
    arg = Args()
    trainer = MultiTrainer(arg)
    # print()
    trainer.train()
    trainer.test()

# preprocess/data/IEMOCAP_V1_order3.npy
