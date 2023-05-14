import copy
import math
import os
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torch.autograd import Function
import torch.optim
from torch.cuda.amp import autocast, GradScaler
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from config import Args
from multiModel import MModel, mmd
from blocks import Chomp1d
from utils import get_newest_file, load_loader, NoamScheduler, Metric, EarlyStopping, accuracy_cal

dataset_name = ['MODMA', 'CASIA2']
num_class = [2, 2]
seq_len = [313, 313]
num_sample = [3549, 2400]
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


class globalDiscriminator(nn.Module):
    def __init__(self, arg):
        super(globalDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=arg.filters, out_channels=16, kernel_size=3, padding="same"),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            Rearrange("N C L -> N (C L)"),
            nn.Linear(arg.seq_len * 16, 1000),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1000, 2),
        )

    def forward(self, x, y):
        x = self.net(x)
        loss = F.cross_entropy(x, y, label_smoothing=0.1)
        correct_num = accuracy_cal(x, y)
        return loss, correct_num


class localDiscriminator(nn.Module):
    def __init__(self, arg):
        super(localDiscriminator, self).__init__()
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
            nn.Linear(1000, 2),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class DDGTrainer:
    def __init__(self, args: Args):
        args.num_class = num_class
        self.optimizer_type = args.optimizer_type
        self.epochs = args.epochs
        self.inner_iter = 34
        self.mmd_step = 6
        self.feature_dim = args.feature_dim
        self.batch_size = args.batch_size
        self.seq_len = args.seq_len
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.best_path = "models/ddg/MODMA_best.pt"
        self.model_path = "models/ddg/MODMA.pt"
        self.pretrain_path = f"models/ddg/pretrain_{dataset_name[1]}.pt"
        self.pretrain_best_path = f"models/ddg/pretrain_{dataset_name[1]}_best.pt"
        self.result_train_path = "results/data/ddg/train.npy"
        self.result_test_path = "results/data/ddg/test.npy"
        self.models = []
        self.record_epochs = []
        self.test_acc = []
        self.D_M, self.D_C, self.MU = 0, 0, 0
        self.d_c, self.d_m = 0, 0
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
    def get_mmd_loss(model, src_model: MModel, hidden1, hidden2, train_batch):
        x1 = model.get_generalFeature(train_batch[0][0].to(device))
        x2 = src_model.get_generalFeature(train_batch[1][0].to(device))
        mini_shape = min([x1.shape[0], x2.shape[0]])
        x1, x2 = x1[:mini_shape], x2[:mini_shape]
        mmd_feature = [hidden1(x1), hidden2(x2)]
        mini_shape = min([x.shape[0] for x in mmd_feature])
        mmd_loss = mmd(mmd_feature[0][:mini_shape].view(mini_shape, -1),
                       mmd_feature[1][:mini_shape].view(mini_shape, -1))
        return mmd_loss

    def get_domain_loss(self, model, src_model: MModel, hidden1, hidden2, gdiscriminator,
                        l1discriminator, l2discriminator, train_batch, p):
        x1 = model.get_generalFeature(train_batch[0][0].to(device))  # target
        x2 = src_model.get_generalFeature(train_batch[1][0].to(device))  # source
        # x1 = hidden1(x1)
        # x2 = hidden2(x2)
        p1 = model.get_softmax(train_batch[0][0].to(device))
        p2 = src_model.get_softmax(train_batch[1][0].to(device))
        # generalFeature = [x1, x2]
        # x = torch.cat(generalFeature, dim=0)
        label = torch.cat([torch.ones(len(train_batch[0][0])), torch.zeros(len(train_batch[1][0]))], dim=0).long()
        alpha = 2. / (1 + np.exp((-10. * p))) - 1
        # x = GRL.apply(x, alpha)
        x1_r, x2_r = GRL.apply(x1, alpha), GRL.apply(x2, alpha)
        x_r = torch.cat([x1_r, x2_r], dim=0)
        gloss, correct_num_domain = gdiscriminator(x_r, label.to(device))
        ft1 = p1[:, 0].reshape((x1_r.shape[0], 1, 1)) * x1_r
        ft2 = p1[:, 1].reshape((x1_r.shape[0], 1, 1)) * x1_r
        fs1 = p2[:, 0].reshape((x2_r.shape[0], 1, 1)) * x2_r
        fs2 = p2[:, 1].reshape((x2_r.shape[0], 1, 1)) * x2_r
        s_domain = torch.zeros(len(train_batch[1][0])).long().to(device)
        t_domain = torch.ones(len(train_batch[0][0])).long().to(device)
        s_out = [l1discriminator(fs1), l2discriminator(fs2)]
        t_out = [l1discriminator(ft1), l2discriminator(ft2)]
        loss_s1, loss_t1 = F.nll_loss(F.log_softmax(s_out[0], dim=1), s_domain), F.nll_loss(
            F.log_softmax(t_out[0], dim=1), t_domain)
        loss_s2, loss_t2 = F.nll_loss(F.log_softmax(s_out[1], dim=1), s_domain), F.nll_loss(
            F.log_softmax(t_out[1], dim=1), t_domain)
        loss_s = loss_s1 + loss_s2
        loss_t = loss_t1 + loss_t2
        tmpd_c = (1 - 2 * (loss_s1 + loss_t1)) + (1 - 2 * (loss_s2 + loss_t2))
        lloss = loss_s + loss_t

        self.d_c = self.d_c + tmpd_c.cpu().item()
        self.d_m = self.d_m + 2 * (1 - 2 * gloss.cpu().item())

        join_loss = (1 - self.MU) * gloss + self.MU * lloss
        return join_loss, correct_num_domain

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

    def pretrain(self):
        model = MModel(arg, index=1)
        model = model.to(device)
        lr = 2e-4
        optimizer = self.get_optimizer(arg, model.parameters(), lr)
        scheduler = self.get_scheduler(optimizer, arg)
        train_num = len(self.loader[1][0].dataset)
        val_num = len(self.loader[1][1].dataset)
        best_val_accuracy = 0
        metric = Metric()
        for epoch in range(50):
            model.train()
            train_acc = 0
            train_loss = 0
            for batch in self.loader[1][0]:
                loss, correct_num = self.train_step(model, optimizer, batch)
                train_acc += correct_num.cpu().numpy()
                train_loss += loss.data.item()
            train_acc /= train_num
            train_loss /= math.ceil(train_num / self.batch_size)
            metric.train_acc.append(train_acc)
            metric.train_loss.append(train_loss)
            print(f"epoch {epoch + 1}: train_acc: {train_acc * 100:.3f}\t train_loss: {train_loss:.4f}")

            model.eval()
            val_acc = 0
            val_loss = 0
            with torch.no_grad():
                for batch in self.loader[1][1]:
                    loss, correct_num = self.val_step(model, batch)
                    val_acc += correct_num.cpu().numpy()
                    val_loss += loss.data.item()
            val_acc /= val_num
            val_loss /= math.ceil(val_num / self.batch_size)
            metric.val_acc.append(val_acc)
            metric.val_loss.append(val_loss)
            print(f"epoch {epoch + 1}: val_acc: {val_acc * 100:.3f}\t val_loss: {val_loss:.4f}")
            scheduler.step()

            if val_acc > best_val_accuracy:
                print(f"val_accuracy improved from {best_val_accuracy} to {val_acc}")
                best_val_accuracy = val_acc
                metric.best_val_acc = [best_val_accuracy, train_acc]
                torch.save(model, self.pretrain_best_path)
                print(f"saving model to {self.pretrain_best_path}")
            elif val_acc == best_val_accuracy:
                if train_acc > metric.best_val_acc[1]:
                    metric.best_val_acc[1] = train_acc
                    print("update train accuracy")
            else:
                print(f"val_accuracy did not improve from {best_val_accuracy}")
            plt.clf()
            plt.plot(metric.train_acc)
            plt.plot(metric.val_acc)
            plt.ylabel("accuracy(%)")
            plt.xlabel("epoch")
            plt.title(f"train accuracy and validation accuracy")
            plt.pause(0.02)
            plt.ioff()  # 关闭画图的窗口

        torch.save(model, self.pretrain_path)
        test_acc = 0
        test_num = len(self.loader[1][2].dataset)
        with torch.no_grad():
            for batch in self.loader[1][2]:
                loss, correct_num = self.test_step(model, batch)
                test_acc += correct_num
        print("final path")
        test_acc = test_acc / test_num
        print(f"test Accuracy:{test_acc * 100:.3f}\t")
        model = torch.load(self.pretrain_best_path)
        test_acc = 0
        with torch.no_grad():
            for batch in self.loader[1][2]:
                loss, correct_num = self.test_step(model, batch)
                test_acc += correct_num
        print("best path")
        test_acc = test_acc / test_num
        print(f"test Accuracy:{test_acc * 100:.3f}\t")

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
        hidden2 = Hidden(arg).to(device)
        parameter = [
            {'params': model.prepare.parameters(), 'lr': arg.lr},
            {'params': model.shareNet.parameters(), 'lr': arg.lr},
            {'params': hidden1.parameters(), 'lr': arg.lr},
            {'params': hidden2.parameters(), 'lr': arg.lr},

        ]
        share_optimizer = self.get_optimizer(arg, parameter, lr=arg.lr)
        share_scheduler = torch.optim.lr_scheduler.StepLR(share_optimizer, step_size=40, gamma=0.3)
        gdiscriminator = globalDiscriminator(arg).to(device)
        gdiscriminator.train()
        l1discriminator = localDiscriminator(arg).to(device)
        l1discriminator.train()
        l2discriminator = localDiscriminator(arg).to(device)
        l2discriminator.train()
        parameter = [
            {'params': model.prepare.parameters(), 'lr': arg.lr},
            {'params': model.shareNet.parameters(), 'lr': arg.lr},
            {'params': hidden1.parameters(), 'lr': arg.lr},
            {"params": hidden2.parameters(), 'lr': arg.lr},
            {"params": gdiscriminator.parameters(), 'lr': arg.lr},
            {'params': l1discriminator.parameters(), 'lr': arg.lr},
            {"params": l2discriminator.parameters(), 'lr': arg.lr}
        ]
        disc_optimizer = self.get_optimizer(arg, parameter, lr=arg.lr)
        disc_scheduler = torch.optim.lr_scheduler.StepLR(disc_optimizer, step_size=40, gamma=0.3)
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
            model.train()
            for batch in self.loader[0][0]:
                loss, correct_num = self.train_step(model, optimizer, batch)
                train_acc += correct_num.cpu().numpy()
                train_loss += loss.data.item()
            if (epoch + 1) % self.mmd_step == 0 and epoch < 61:
                m_loss = []
                self.d_m, self.d_c = 0, 0
                ''' update mu per epoch '''
                if self.D_M == 0 and self.D_C == 0 and self.MU == 0:
                    self.MU = 0.5
                else:
                    self.D_M = self.D_M / num_sample[1]
                    self.D_C = self.D_C / num_sample[1]
                    self.MU = 1 - self.D_M / (self.D_M + self.D_C)

                for step in range(self.inner_iter):
                    train_batch = []
                    for i in range(dataset_num):
                        try:
                            batch = next(train_iter[i])
                        except StopIteration:
                            train_iter[i] = iter(self.loader[i][0])
                            batch = next(train_iter[i])
                        train_batch.append(batch)
                    mmd_loss = self.get_mmd_loss(model, src_model, hidden1, hidden2, train_batch)
                    m_loss.append(mmd_loss.data.item())
                    share_optimizer.zero_grad()
                    mmd_loss.backward()
                    share_optimizer.step()
                    p = epoch / self.epochs
                    domain_loss, correct_num = self.get_domain_loss(model, src_model, hidden1, hidden2, gdiscriminator,
                                                                    l1discriminator, l2discriminator, train_batch, p)

                    disc_optimizer.zero_grad()
                    domain_loss.backward()
                    disc_optimizer.step()
                    domain_acc += correct_num.cpu().numpy()
                    domain_num += len(train_batch[1][0])
                    domain_num += len(train_batch[0][0])
                self.D_M = np.copy(self.d_m).item()
                self.D_C = np.copy(self.d_c).item()
                print(self.MU)
                domain_acc = domain_acc / domain_num
                print(f"domain accuracy: {domain_acc:.3f}")
                print(np.mean(m_loss))

            scheduler.step()
            share_scheduler.step()
            disc_scheduler.step()
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
    trainer = DDGTrainer(arg)
    # trainer.pretrain()
    trainer.train()
    trainer.test()
    print(trainer.test_acc)
    # for m in trainer.models:
    #     print(m.prepare.net[0].bias[0:9])
