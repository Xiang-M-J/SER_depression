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
from multiModel import MModel, mmd, Discriminator
from utils import get_newest_file, load_loader, NoamScheduler, Metric, EarlyStopping, accuracy_cal, load_dataset
from blocks import Temporal_Aware_Block

dataset_name = ['MODMA', 'CASIA']
num_class = [2, 6]
seq_len = [313, 313]
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
        self.act = nn.LeakyReLU()
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
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Encoder(nn.Module):
    def __init__(self, arg: Args):
        super(Encoder, self).__init__()
        self.conv1 = ConvBlock(arg.filters, 64, kernel_size=3, pool_size=2)
        self.conv2 = ConvBlock(64, 128, kernel_size=3, pool_size=2)
        self.conv3 = ConvBlock(128, 256, kernel_size=3, pool_size=3)

        # self.upsample1 = UpsampleBlock(256, 128, scale_factor=2)
        # self.upsample2 = UpsampleBlock(128, 64, scale_factor=2)
        # self.upsample3 = UpsampleBlock(64, arg.filters, scale_factor=2, size=arg.seq_len)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.upsample1(x)
        # x = self.upsample2(x)
        # x = self.upsample3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, arg: Args):
        super(Decoder, self).__init__()
        # self.conv1 = ConvBlock(arg.filters, 64, kernel_size=3, pool_size=2)
        # self.conv2 = ConvBlock(64, 128, kernel_size=3, pool_size=2)
        # self.conv3 = ConvBlock(128, 256, kernel_size=3, pool_size=2)
        self.upsample1 = UpsampleBlock(256, 128, scale_factor=3)
        self.upsample2 = UpsampleBlock(128, 64, scale_factor=2)
        self.upsample3 = UpsampleBlock(64, arg.filters, scale_factor=2, size=arg.seq_len)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        return x


class Fusion(nn.Module):
    def __init__(self, arg: Args):
        super(Fusion, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(arg.filters, arg.filters, kernel_size=3, padding="same"),
            nn.BatchNorm1d(arg.filters),
            nn.LeakyReLU(),
        )
        self.l1 = nn.Sequential(
            nn.Linear(arg.seq_len, arg.seq_len),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        for _ in range(3):
            x = self.conv1(x)
        for _ in range(2):
            x = self.l1(x)
        return x


class MultiTrainer:
    def __init__(self, args: Args):
        args.num_class = num_class
        self.optimizer_type = args.optimizer_type
        self.epochs = args.epochs
        self.iteration = 4200
        self.inner_iter = 50
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
        self.random_seed = args.random_seed
        args.spilt_rate = split_rate
        self.loader = []
        # batch_size = [48, 16]

    def get_loader(self, batch_size):
        loader = []
        for i in range(dataset_num):
            train_loader, val_loader, test_loader = \
                load_loader(dataset_name[i], split_rate, batch_size[i], self.random_seed, version='V1', order=3)
            loader.append([train_loader, val_loader, test_loader])
        return loader

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

    def pretext_train_step(self, model, optimizer, batch):
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

    def train_step(self, model, optimizer, batch):
        x, y = self.get_data(batch[0])
        fake_x, fake_y = self.get_data(batch[1])
        mini_shape = fake_x.shape[0]
        # 没有fusion
        x_f = model.prepare(x[:mini_shape])
        x_b = model.prepare(torch.flip(x[:mini_shape], dims=[-1]))
        x_1, _, _ = model.shareNet(x_f, x_b)
        # 有fusion
        # x_1, fake_x, _ = model.shareNet(fake_x[:mini_shape], fake_x[:mini_shape])

        x_2, _ = model.specialNet[0](fake_x)
        fake_x = model.specialNet[1](torch.cat([x_1, x_2], dim=-1))
        fake_x = model.specialNet[2](fake_x)
        loss = F.cross_entropy(fake_x, fake_y)
        correct_num = accuracy_cal(fake_x, fake_y)
        optimizer.zero_grad()
        if use_amp:
            with autocast():
                loss, correct_num = model(x, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_, correct_num_ = model(x, y)
            loss += loss_
            correct_num += correct_num_
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
    def fake_step(model, src_model, encoder, decoder, fusion, train_batch):
        x1, x2 = train_batch[0][0], train_batch[1][0]
        mini_shape = min(x1.shape[0], x2.shape[0])
        x1, x2 = x1[:mini_shape].to(device), x2[:mini_shape].to(device)
        x_f = model.prepare(x1)
        x_b = model.prepare(torch.flip(x1, dims=[-1]))
        _, x1, _ = model.shareNet(x_f, x_b)
        x2 = src_model.get_generalFeature(x2)
        h_x1, h_x2 = encoder(x1), encoder(x2)
        r_x1 = decoder(h_x1)
        r_xf = fusion(r_x1)
        ae_loss1 = F.mse_loss(x1, r_x1)
        ae_loss2 = F.mse_loss(x_f, r_xf)
        mmd_loss = mmd(h_x1.view(mini_shape, -1), h_x2.view(mini_shape, -1))
        return mmd_loss + ae_loss1 + ae_loss2

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

    def pretext(self):
        loader = self.get_loader([64, 64])
        # 预训练
        # train_acc = 0
        # val_acc = 0
        # model = MModel(arg, seq_len=seq_len[0], index=0).to(device)
        # optimizer = self.get_optimizer(arg, model.parameters(), lr=arg.lr)
        # scheduler = self.get_scheduler(optimizer, arg)
        # for _ in range(80):
        #     model.train()
        #     for batch in loader[0][0]:
        #         loss, correct_num = self.pretext_train_step(model, optimizer, batch)
        #         train_acc += correct_num.cpu().numpy()
        #
        #     model.eval()
        #     with torch.no_grad():
        #         for batch in loader[0][1]:
        #             loss, correct_num = self.val_step(model, batch)
        #             val_acc += correct_num.cpu().numpy()
        #     val_acc = val_acc / int(num_sample[0] * split_rate[1])
        #     train_acc = train_acc / int(num_sample[0] * split_rate[0])
        #     print(f"train_acc: {train_acc}")
        #     print(f"val_acc: {val_acc}")
        #     train_acc = 0
        #     val_acc = 0
        #     scheduler.step()
        # torch.save(model, "models/Multi/MODMA_pretrain.pt")

        train_iter = [iter(loader[i][0]) for i in range(dataset_num)]
        # 直接加载
        model = torch.load("models/Multi/MODMA_pretrain.pt")
        model.eval()
        src_model = torch.load(self.pretrain_path)
        src_model.eval()
        encoder = Encoder(arg).to(device)
        decoder = Decoder(arg).to(device)
        fusion = Fusion(arg).to(device)
        parameter = [
            {'params': encoder.parameters(), 'lr': 1e-3},
            {'params': decoder.parameters(), 'lr': 1e-3},
            {'params': fusion.parameters(), 'lr': 1e-3}
        ]
        # fake_optimizer = self.get_optimizer(arg, parameter, lr=arg.lr)
        fake_optimizer = torch.optim.AdamW(parameter, lr=1e-3)
        fake_scheduler = torch.optim.lr_scheduler.StepLR(fake_optimizer, step_size=10, gamma=0.3)
        m_loss = []
        for step in range(3000):
            encoder.train()
            decoder.train()
            fusion.train()
            train_batch = []
            for i in range(dataset_num):
                try:
                    batch = next(train_iter[i])
                except StopIteration:
                    train_iter[i] = iter(loader[i][0])
                    batch = next(train_iter[i])
                train_batch.append(batch)
            loss = self.fake_step(model, src_model, encoder, decoder, fusion, train_batch)
            m_loss.append(loss.data.item())
            fake_optimizer.zero_grad()
            loss.backward()
            fake_optimizer.step()
            if step % 50 == 0:
                print(f"step: {step}")
                print(np.mean(m_loss))
                m_loss = []
                fake_scheduler.step()
        torch.save(encoder, "../models/Multi/Encoder_data.pt")
        torch.save(decoder, "../models/Multi/Decoder_data.pt")
        torch.save(fusion, "../models/Multi/Fusion_data.pt")

    def train(self):
        loader = self.get_loader([64, 32])
        mini_iter = min([len(loader[i][0]) for i in range(dataset_num)])
        train_iter = [iter(loader[i][0]) for i in range(dataset_num)]
        src_model = torch.load(self.pretrain_path)
        src_model.eval()
        model = MModel(arg, seq_len=seq_len[0], index=0).to(device)
        optimizer = self.get_optimizer(arg, model.parameters(), lr=arg.lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50)
        # encoder = Encoder(arg).to(device)
        # decoder = Decoder(arg).to(device)
        encoder = torch.load("models/Multi/Encoder_data.pt").to(device)
        encoder.eval()
        decoder = torch.load("models/Multi/Decoder_data.pt").to(device)
        decoder.eval()
        fusion = torch.load("models/Multi/Fusion_data.pt").to(device)
        fusion.eval()
        parameter = [
            {'params': model.prepare.parameters(), 'lr': arg.lr},
            {'params': model.shareNet.parameters(), 'lr': arg.lr},
            # {'params': encoder.parameters(), 'lr': arg.lr},
            # {'params': decoder.parameters(), 'lr': arg.lr}
        ]
        fake_optimizer = self.get_optimizer(arg, parameter, lr=arg.lr)
        fake_scheduler = self.get_scheduler(fake_optimizer, arg)

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
        for step in range(self.iteration):
            model.train()
            m_loss = []
            train_batch = []
            for i in range(dataset_num):
                try:
                    batch = next(train_iter[i])
                except StopIteration:
                    train_iter[i] = iter(loader[i][0])
                    batch = next(train_iter[i])
                train_batch.append(batch)

            # for _ in range(self.inner_iter):
            #     loss = self.fake_step(model, src_model, encoder, decoder, train_batch)
            #     m_loss.append(loss.data.item())
            #     fake_optimizer.zero_grad()
            #     loss.backward()
            #     fake_optimizer.step()
            # print(np.mean(m_loss))
            mini_shape = min(train_batch[0][0].shape[0], train_batch[1][0].shape[0])
            fake_x = (decoder(encoder(train_batch[1][0][:mini_shape].to(device))))
            fake_y = train_batch[0][1][:mini_shape]
            train_batch[1] = [fake_x, fake_y]

            loss, correct_num = self.train_step(model, optimizer, train_batch)
            train_acc += correct_num.cpu().numpy()
            train_loss += loss.data.item()
            train_num += len(train_batch[0][0])
            train_num += len(train_batch[1][0])

            if (step + 1) % mini_iter == 0:
                print(f"step: {step + 1}")
                # scheduler.step()
                train_acc = train_acc / train_num
                train_loss = train_loss / mini_iter
                metric.train_acc.append(train_acc)
                metric.train_loss.append(train_loss)
                print(f"MODMA: train Loss:{train_loss:.4f}\t train Accuracy:{train_acc * 100:.3f}\t")
                model.eval()
                with torch.no_grad():
                    for batch in loader[0][1]:
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
        loader = self.get_loader(batch_size=[64, 64])
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
            for batch in loader[0][2]:
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
            for batch in loader[0][2]:
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
    trainer.train()
    trainer.test()