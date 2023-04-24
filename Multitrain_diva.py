import math
import os

import numpy as np
import torch.nn as nn
import torch.optim
from matplotlib import pyplot as plt
from torch.autograd import Function
from torch.cuda.amp import autocast, GradScaler
from DIVA import DIVA, qzd, qzx, qzy
from config import Args
from multiModel import MModel, Discriminator, mmd
from utils import get_newest_file, load_loader, NoamScheduler, Metric
import torch.distributions as dist

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
        self.fc1 = nn.Sequential(
            nn.Conv1d(arg.filters, arg.filters, kernel_size=1, padding="same"),
            nn.BatchNorm1d(arg.filters),
            nn.ReLU(),
            nn.Conv1d(arg.filters, arg.filters, kernel_size=3, padding="same"),
            nn.BatchNorm1d(arg.filters),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Conv1d(arg.filters, arg.filters, kernel_size=3, padding="same"),
            nn.BatchNorm1d(arg.filters),
            nn.ReLU(),
            nn.Conv1d(arg.filters, arg.filters, kernel_size=3, padding="same"),
            nn.BatchNorm1d(arg.filters),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Conv1d(arg.filters, arg.filters, kernel_size=3, padding="same"),
            nn.BatchNorm1d(arg.filters),
            nn.ReLU(),
            nn.Conv1d(arg.filters, arg.filters, kernel_size=1, padding="same"),
            nn.BatchNorm1d(arg.filters),
            nn.ReLU()
        )


    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class MultiTrainer:
    def __init__(self, args: Args):
        args.num_class = num_class
        # args.seq_len = seq_len
        self.optimizer_type = args.optimizer_type
        self.epochs = args.epochs
        self.iteration = 5000
        self.inner_iter = 20
        self.mmd_step = 8
        self.feature_dim = args.feature_dim
        self.batch_size = args.batch_size
        self.seq_len = args.seq_len
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.best_path = "models/Multi/MODMA_best.pt"
        self.encoder_final_path = f"models/Multi/encoder.pt"
        self.model_path = "models/Multi/MODMA.pt"
        self.pretrain_path = "models/Multi/pretrain.pt"
        self.pretrain_best_path = "models/Multi/pretrain_best.pt"
        self.pretext_epochs = 50
        args.spilt_rate = split_rate
        self.loader = []
        for i in range(dataset_num):
            train_loader, val_loader, test_loader = \
                load_loader(dataset_name[i], split_rate, args.batch_size, args.random_seed, version='V1', order=3)
            self.loader.append([train_loader, val_loader, test_loader])
        self.qzd = qzd(args.filters, 256).to(device)
        self.qzx = qzx(args.filters, 256).to(device)
        self.qzy = qzy(args.filters, 256).to(device)

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
    def pseudo_label(batch_size, seq_len):
        label = torch.zeros([batch_size, 2, seq_len])
        num = int(batch_size / 2)
        label[:num, 0, :] = 1.
        label[num:, 1, :] = 1.
        return label[torch.randperm(label.size(0))].type(torch.FloatTensor)

    @staticmethod
    def get_data(batch):
        x, y = batch
        x, y = x.to(device), y.to(device)
        return x, y

    def get_mmd_loss(self, model, src_model, hidden, train_batch, sigma):

        x1 = model.get_generalFeature(train_batch[0][0].to(device))
        x2 = model.get_generalFeature(train_batch[1][0].to(device))
        mini_shape = min(x1.shape[0], x2.shape[0])
        x1, x2 = x1[:mini_shape], x2[:mini_shape]
        x1 = hidden(x1)
        x2 = hidden(x2)
        zd_x1_loc, zd_x1_scale = self.qzd(x1)
        zd_x2_loc, zd_x2_scale = self.qzd(x2)
        zx_x1_loc, zx_x1_scale = self.qzx(x1)
        zx_x2_loc, zx_x2_scale = self.qzx(x2)
        zy_x1_loc, zy_x1_scale = self.qzy(x1)
        zy_x2_loc, zy_x2_scale = self.qzy(x2)
        q_x1_d, q_x2_d = dist.Normal(zd_x1_loc, zd_x1_scale), dist.Normal(zd_x2_loc, zd_x2_scale)
        q_x1_x, q_x2_x = dist.Normal(zx_x1_loc, zx_x1_scale), dist.Normal(zx_x2_loc, zx_x2_scale)
        q_x1_y, q_x2_y = dist.Normal(zy_x1_loc, zy_x1_scale), dist.Normal(zy_x2_loc, zy_x2_scale)
        zd_x1, zd_x2 = q_x1_d.rsample(), q_x2_d.rsample()
        zx_x1, zx_x2 = q_x1_x.rsample(), q_x2_x.rsample()
        zy_x1, zy_x2 = q_x1_y.rsample(), q_x2_y.rsample()
        zd_loss = mmd(zd_x1, zd_x2)
        zx_loss = mmd(zx_x1, zx_x2)
        zy_loss = mmd(zy_x1, zy_x2)
        loss = zd_loss / (sigma[0]**2) + zx_loss / (sigma[1]**2) + zy_loss / (sigma[2]**2)\
               + torch.log(sigma[0]) + torch.log(sigma[1]) + torch.log(sigma[2])
        return loss

    @staticmethod
    def get_domain_loss(model, src_model, hidden, discriminator, train_batch, p):
        x1 = model.get_generalFeature(train_batch[0][0].to(device))
        x2 = src_model.get_generalFeature(train_batch[1][0].to(device))
        generalFeature = [hidden(x1), hidden(x2)]
        x = torch.cat(generalFeature, dim=0)
        label = torch.cat([torch.zeros(len(train_batch[0][0])), torch.ones(len(train_batch[1][0]))], dim=0).long()
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

    def train(self):
        arg.step_size = 30
        arg.gamma = 0.3
        mini_iter = min([len(self.loader[i][0]) for i in range(dataset_num)])
        train_iter = [iter(self.loader[i][0]) for i in range(dataset_num)]
        src_model = torch.load(self.pretrain_path)
        src_model.eval()
        model = MModel(arg, seq_len=seq_len[0], index=0).to(device)
        optimizer = self.get_optimizer(arg, model.parameters(), lr=arg.lr)
        scheduler = self.get_scheduler(optimizer, arg)
        hidden1 = Hidden(arg).to(device)
        hidden1.train()
        hidden2 = Hidden(arg).to(device)
        hidden2.train()
        sigma1 = torch.ones((1,)).to(device)
        sigma2 = torch.ones((1,)).to(device)
        sigma3 = torch.ones((1,)).to(device)
        sigma1.requires_grad = True
        sigma2.requires_grad = True
        sigma3.requires_grad = True
        parameter = [
            {'params': model.prepare.parameters(), 'lr': arg.lr},
            {'params': model.shareNet.parameters(), 'lr': arg.lr},
            {'params': hidden1.parameters(), 'lr': arg.lr},
            {'params': sigma1, 'lr': arg.lr},
            {'params': sigma2, 'lr': arg.lr},
            {'params': sigma3, 'lr': arg.lr}
        ]
        # share_optimizer = torch.optim.RMSprop(parameter, lr=arg.lr, weight_decay=arg.weight_decay)
        share_optimizer = self.get_optimizer(arg, parameter, arg.lr)
        share_scheduler = self.get_scheduler(share_optimizer, arg)

        # encoder = torch.load(self.encoder_final_path)
        # encoder.eval()
        # tgt_encoder = Encoder(arg).to(device)
        # tgt_encoder.load_state_dict(encoder.state_dict())

        discriminator = Discriminator(arg).to(device)
        discriminator.train()
        parameter = [
            {'params': model.prepare.parameters(), 'lr': arg.lr},
            {'params': model.shareNet.parameters(), 'lr': arg.lr},
            {"params": hidden1.parameters(), 'lr': arg.lr},
            {"params": discriminator.parameters(), 'lr': arg.lr}
        ]
        disc_optimizer = self.get_optimizer(arg, parameter, arg.lr)
        disc_scheduler = self.get_scheduler(disc_optimizer, arg)

        parameter = [
            {'params': model.specialNet.parameters(), 'lr': arg.lr},
            # {'params': model.shareNet.parameters(), 'lr': arg.lr},
            # {'params': model.prepare.parameters(), 'lr': arg.lr},
            # {'params': hidden2.parameters(), 'lr': arg.lr},
        ]
        special_optimizer = self.get_optimizer(arg, parameter, arg.lr)
        special_scheduler = self.get_scheduler(special_optimizer, arg)
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
            for batch in self.loader[0][0]:
                loss, correct_num = self.train_step(model, optimizer, batch)
                train_acc += correct_num.cpu().numpy()
                train_loss += loss.data.item()

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
                    mmd_loss = self.get_mmd_loss(model, src_model, hidden1, train_batch, [sigma1, sigma2, sigma3])
                    m_loss.append(mmd_loss.data.item())
                    share_optimizer.zero_grad()
                    mmd_loss.backward()
                    share_optimizer.step()

                    # p = epoch / self.epochs
                    # domain_loss, correct_num = self.get_domain_loss(model, src_model, hidden1, discriminator,
                    #                                                 train_batch, p)
                    # disc_optimizer.zero_grad()
                    # domain_loss.backward()
                    # disc_optimizer.step()
                    domain_acc += correct_num.cpu().numpy()
                    domain_num += len(train_batch[1][0])
                    domain_num += len(train_batch[0][0])

                    # special_loss = self.get_special_loss(model, src_model, hidden2, train_batch)
                    # special_optimizer.zero_grad()
                    # special_loss.backward()
                    # special_optimizer.step()

                domain_acc = domain_acc / domain_num
                print(f"domain accuracy: {domain_acc:.3f}")

                print(np.mean(m_loss))

            scheduler.step()
            share_scheduler.step()
            disc_scheduler.step()
            special_scheduler.step()
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
    # trainer.pretext()
    # trainer.pretrain()
    trainer.train()
    trainer.test()
