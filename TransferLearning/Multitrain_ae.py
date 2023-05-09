import math
import os
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torch.autograd import Function
import torch.optim
from torch.cuda.amp import autocast, GradScaler
from einops.layers.torch import Rearrange

from config import Args
from multiModel import MModel, mmd, Discriminator
from utils import get_newest_file, load_loader, NoamScheduler, Metric, EarlyStopping, accuracy_cal

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


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, pool_size):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding="same")
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool1d(pool_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_dim, out_dim, scale_factor, size=None):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, padding="same")
        if size is None:
            self.upsample = nn.Upsample(scale_factor=scale_factor)
        else:
            self.upsample = nn.Upsample(size=size)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Encoder(nn.Module):
    def __init__(self, arg: Args):
        super(Encoder, self).__init__()
        # self.conv1 = ConvBlock(arg.feature_dim, 64, 3, 2)
        # self.conv2 = ConvBlock(64, 64, 3, 2)
        # self.conv3 = ConvBlock(64, 128, 3, 4)
        # self.conv4 = ConvBlock(128, 256, 3, 4)
        # self.l1 = nn.Linear(arg.seq_len, 128),
        # self.l2 = nn.Linear(128, 64),
        # self.l3 = nn.Linear(64, 32),
        # self.l4 = nn.Linear(32, 16),
        # self.l5 = nn.Linear(16, 8)
        self.net = nn.Sequential(
            nn.Linear(arg.seq_len, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 32),
            # nn.Linear(32, 16),
            # nn.Linear(16, 8)
        )

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        x = self.net(x)
        return x


class Decoder(nn.Module):
    def __init__(self, arg: Args):
        super(Decoder, self).__init__()
        # self.upsample1 = UpsampleBlock(256, 128, 4)
        # self.upsample2 = UpsampleBlock(128, 64, 4)
        # self.upsample3 = UpsampleBlock(64, 64, 2)
        # self.upsample4 = UpsampleBlock(64, arg.feature_dim, 2, size=arg.seq_len)
        self.net = nn.Sequential(
            # nn.Linear(8, 16),
            # nn.Linear(16, 32),
            nn.Linear(32, 64),
            nn.Linear(64, 128),
            nn.Linear(128, arg.seq_len)
        )

    def forward(self, x):
        # x = self.upsample1(x)
        # x = self.upsample2(x)
        # x = self.upsample3(x)
        # x = self.upsample4(x)
        x = self.net(x)
        return x


class Hidden(nn.Module):
    def __init__(self, arg: Args):
        super(Hidden, self).__init__()
        self.conv1 = ConvBlock(arg.filters, 64, kernel_size=3, pool_size=2)
        self.conv2 = ConvBlock(64, 128, kernel_size=3, pool_size=2)
        self.conv3 = ConvBlock(128, 256, kernel_size=3, pool_size=2)
        self.upsample1 = UpsampleBlock(256, 128, scale_factor=2)
        self.upsample2 = UpsampleBlock(128, 64, scale_factor=2)
        self.upsample3 = UpsampleBlock(64, arg.filters, scale_factor=2, size=arg.seq_len)
        #
        # self.fc1 = nn.Sequential(
        #     nn.Conv1d(arg.filters, arg.filters, kernel_size=1, padding="same"),
        #     nn.BatchNorm1d(arg.filters),
        #     nn.ReLU(),
        #     nn.Conv1d(arg.filters, arg.filters, kernel_size=3, padding="same"),
        #     nn.BatchNorm1d(arg.filters),
        #     nn.ReLU()
        # )
        # self.fc21 = nn.Sequential(nn.Conv1d(arg.filters, arg.filters, kernel_size=3, padding="same"))
        # self.fc22 = nn.Sequential(nn.Conv1d(arg.filters, arg.filters, kernel_size=3, padding="same"))

        # torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        # torch.nn.init.xavier_uniform_(self.fc1[3].weight)
        # torch.nn.init.xavier_uniform_(self.fc21[0].weight)
        # self.fc21[0].bias.data.zero_()
        # torch.nn.init.xavier_uniform_(self.fc22[0].weight)
        # self.fc22[0].bias.data.zero_()

        # self.fc1 = nn.Sequential(
        #     nn.Conv1d(arg.filters, arg.filters, kernel_size=1, padding="same"),
        #     nn.BatchNorm1d(arg.filters),
        #     nn.ReLU(),
        #     nn.Conv1d(arg.filters, arg.filters, kernel_size=3, padding="same"),
        #     nn.BatchNorm1d(arg.filters),
        #     nn.ReLU()
        # )
        # self.fc2 = nn.Sequential(
        #     nn.Conv1d(arg.filters, arg.filters, kernel_size=3, padding="same"),
        #     nn.BatchNorm1d(arg.filters),
        #     nn.ReLU(),
        #     nn.Conv1d(arg.filters, arg.filters, kernel_size=3, padding="same"),
        #     nn.BatchNorm1d(arg.filters),
        #     nn.ReLU()
        # )
        # self.fc3 = nn.Sequential(
        #     nn.Conv1d(arg.filters, arg.filters, kernel_size=3, padding="same"),
        #     nn.BatchNorm1d(arg.filters),
        #     nn.ReLU(),
        #     nn.Conv1d(arg.filters, arg.filters, kernel_size=1, padding="same"),
        #     nn.BatchNorm1d(arg.filters),
        #     nn.ReLU()
        # )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)

        # x = self.fc1(x)
        # x = self.fc21(x)
        # x = self.fc22(x)

        # x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        return x


class MultiTrainer:
    def __init__(self, args: Args):
        args.num_class = num_class
        # args.seq_len = seq_len
        self.optimizer_type = args.optimizer_type
        self.epochs = args.epochs
        self.iteration = 5000
        self.inner_iter = 20
        self.mmd_step = 6
        self.feature_dim = args.feature_dim
        self.batch_size = args.batch_size
        self.seq_len = args.seq_len
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.best_path = "../models/Multi/MODMA_best.pt"
        self.encoder_final_path = f"../models/Multi/Encoder.pt"
        self.model_path = "../models/Multi/MODMA.pt"
        self.pretrain_path = "../models/Multi/pretrain.pt"
        self.pretrain_best_path = "../models/Multi/pretrain_best.pt"
        self.pretext_epochs = 50
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

    @staticmethod
    def get_mmd_loss(model, src_model, hidden, train_batch):
        x1 = model.get_generalFeature(train_batch[0][0].to(device))
        x2 = src_model.get_generalFeature(train_batch[1][0].to(device))
        mmd_feature = [hidden(x1), hidden(x2)]
        mini_shape = min([x.shape[0] for x in mmd_feature])
        mmd_loss = mmd(mmd_feature[0][:mini_shape].view(mini_shape, -1),
                       mmd_feature[1][:mini_shape].view(mini_shape, -1))
        return mmd_loss

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
        arg = Args()
        arg.num_class = num_class
        arg.epochs = 60
        arg.step_size = 10
        arg.gamma = 0.3
        model = MModel(arg, index=1)
        model = model.to(device)
        lr = 2e-4

        optimizer = self.get_optimizer(arg, model.parameters(), lr)
        scheduler = self.get_scheduler(optimizer, arg)
        train_num = len(self.loader[1][0].dataset)
        val_num = len(self.loader[1][1].dataset)
        best_val_accuracy = 0
        metric = Metric()
        earlystop = EarlyStopping(patience=5)
        for epoch in range(self.epochs):
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
            if earlystop(val_acc):
                break

        torch.save(model, self.pretrain_path)
        # model = torch.load(self.pretrain_path)
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

    def get_fake(self, encoder, generator, x_no):
        _, z_p, _ = encoder[0](x_no, x_no)
        y_p = self.pseudo_label(x_no.shape[0], x_no.shape[-1]).to(device)  # 伪标签
        fake_x = generator[0:3](torch.cat([y_p, z_p], dim=1))
        _, fake_x, _ = generator[3](fake_x, fake_x)
        return y_p, fake_x

    def pretext(self):
        model = torch.load(self.pretrain_path).to(device)
        encoder = Encoder(arg).to(device)
        decoder = Decoder(arg).to(device)
        criterion = nn.MSELoss().to(device)
        ae_parameter = [
            {"params": encoder.parameters()},
            {"params": decoder.parameters()}
        ]
        ae_optimizer = self.get_optimizer(arg, ae_parameter, 4e-4)
        encoder.train()
        decoder.train()
        model.eval()
        for epoch in range(100):
            # 记录变量置0
            ae_loss = []
            for batch in self.loader[1][0]:
                x, _ = self.get_data(batch)
                # 训练AE
                with torch.no_grad():
                    x = model.get_generalFeature(x)
                h = encoder(x)
                r = decoder(h)
                loss = criterion(r, x)
                ae_optimizer.zero_grad()
                loss.backward()
                ae_optimizer.step()
                ae_loss.append(loss.data.item())
            print(f"epoch{epoch}: AE Loss: {np.mean(ae_loss)}")
        torch.save(encoder, self.encoder_final_path)

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
        hidden = Hidden(arg).to(device)
        parameter = [
            {'params': model.prepare.parameters(), 'lr': arg.lr},
            {'params': model.shareNet.parameters(), 'lr': arg.lr},
            {'params': hidden.parameters(), 'lr': arg.lr}
        ]
        share_optimizer = self.get_optimizer(arg, parameter, lr=arg.lr)
        share_scheduler = torch.optim.lr_scheduler.StepLR(share_optimizer, step_size=40, gamma=0.3)
        # encoder = torch.load(self.encoder_final_path)
        # encoder.eval()
        # tgt_encoder = Encoder(arg).to(device)
        # tgt_encoder.load_state_dict(encoder.state_dict())

        discriminator = Discriminator(arg).to(device)
        discriminator.train()
        parameter = [
            {'params': model.prepare.parameters(), 'lr': arg.lr},
            {'params': model.shareNet.parameters(), 'lr': arg.lr},
            {"params": hidden.parameters(), 'lr': arg.lr},
            {"params": discriminator.parameters(), 'lr': arg.lr}
        ]
        disc_optimizer = self.get_optimizer(arg, parameter, lr=arg.lr)
        # disc_scheduler = self.get_scheduler(disc_optimizer, arg)
        disc_scheduler = torch.optim.lr_scheduler.StepLR(disc_optimizer, step_size=40, gamma=0.3)

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

            if (epoch + 1) % self.mmd_step == 0 and epoch < 96:
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
                    mmd_loss = self.get_mmd_loss(model, src_model, hidden, train_batch)
                    m_loss.append(mmd_loss.data.item())
                    share_optimizer.zero_grad()
                    mmd_loss.backward()
                    share_optimizer.step()
                    p = epoch / self.epochs
                    domain_loss, correct_num = self.get_domain_loss(model, src_model, hidden, discriminator,
                                                                    train_batch, p)
                    disc_optimizer.zero_grad()
                    domain_loss.backward()
                    disc_optimizer.step()
                    domain_acc += correct_num.cpu().numpy()
                    domain_num += len(train_batch[1][0])
                    domain_num += len(train_batch[0][0])
                domain_acc = domain_acc / domain_num
                print(f"domain accuracy: {domain_acc:.3f}")

                print(np.mean(m_loss))

            scheduler.step()
            share_scheduler.step()
            disc_scheduler.step()
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
        np.save("../results/data/Multi/MODMA.npy", metric.item())
        torch.save(model, self.model_path)

    def test(self, path=None):

        if path is None:
            path = get_newest_file("../models/Multi/")
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
        np.save("../results/data/Multi/test.npy", metric.item())


if __name__ == "__main__":
    arg = Args()
    trainer = MultiTrainer(arg)
    # trainer.pretext()
    # trainer.pretrain()
    trainer.train()
    trainer.test()
