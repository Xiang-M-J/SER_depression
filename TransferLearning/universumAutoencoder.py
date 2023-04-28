# 效果一般般 目标域验证集准确率55%（四分类）
import math
import random
import os
import numpy as np
import torch.nn as nn
import torch
from einops.layers.torch import Rearrange
from torch.utils.data.dataloader import DataLoader
from config import Args
from utils import load_loader, Metric, cal_seq_len, accuracy_cal

device = "cuda" if torch.cuda.is_available() else "cpu"


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
        self.conv1 = ConvBlock(arg.feature_dim, 64, 15, 2)
        self.conv2 = ConvBlock(64, 64, 3, 4)
        self.conv3 = ConvBlock(64, 128, 3, 4)
        self.conv4 = ConvBlock(128, 256, 3, 4)
        # self.final = Rearrange("N C L -> N (C L)")

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.final(x)
        return x


class Decoder(nn.Module):
    def __init__(self, arg: Args):
        super(Decoder, self).__init__()
        self.upsample1 = UpsampleBlock(256, 128, 4)
        self.upsample2 = UpsampleBlock(128, 64, 4)
        self.upsample3 = UpsampleBlock(64, 64, 4)
        self.upsample4 = UpsampleBlock(64, arg.feature_dim, 2, size=188)

    def forward(self, x):
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)

        return x


class Prepare(nn.Module):
    def __init__(self, arg: Args):
        super(Prepare, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(arg.feature_dim, arg.filters, kernel_size=1, padding="same"),
            nn.BatchNorm1d(arg.filters),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Universum(nn.Module):
    def __init__(self, arg: Args):
        super(Universum, self).__init__()
        self.in_proj = Rearrange("N C L -> N (C L)")
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 4)
        self.act = nn.LeakyReLU()
        self.out_proj = nn.Sigmoid()

    def forward(self, x):
        x = self.in_proj(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.act(x)
        x = self.linear3(x)
        # x = self.out_proj(x)
        return x


class Classifier(nn.Module):
    def __init__(self, arg: Args):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            Rearrange("N C L -> N (C L)"),
            nn.Linear(256, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(100, 4),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class UAETrainer:
    def __init__(self, arg: Args, src_dataset_name, tgt_dataset_name):
        self.src_dt = src_dataset_name
        self.tgt_dt = tgt_dataset_name
        self.src_train, self.src_val = load_loader(self.src_dt, [0.8, 0.2], arg.batch_size, arg.random_seed, "V1")
        self.tgt_train, self.tgt_val = load_loader(self.tgt_dt, [0.8, 0.2], arg.batch_size, arg.random_seed, "V1")
        self.mini_seq_len = min(self.src_train.dataset.dataset.x.shape[-1], self.tgt_train.dataset.dataset.x.shape[-1])
        self.lr = arg.lr
        self.weight_decay = arg.weight_decay
        self.batch_size = arg.batch_size
        self.feature_dim = arg.feature_dim
        self.pretext_epochs = 50
        self.iteration = 10000
        self.fine_tune_epochs = 20
        self.fine_tune_num = 300
        self.encoder_path = f"../models/universum/encoder.pt"
        self.universum_path = "../models/universum/universum.pt"
        arg.seq_len = self.mini_seq_len
        self.arg = arg
        self.criterion = nn.CrossEntropyLoss()
        self.metric = Metric()

    def get_data(self, batch):
        x, y = batch
        x = x[:, :, :self.mini_seq_len]
        x, y = x.to(device), y.to(device)
        return x, y

    def train(self):
        C = 1
        C_u = 5
        epsilon = 0.001
        xi = 1
        prepare = Prepare(self.arg).to(device)
        encoder = Encoder(self.arg).to(device)
        decoder = Decoder(self.arg).to(device)
        universum = Universum(self.arg).to(device)
        classifier = Classifier(self.arg).to(device)
        src_train_iter, tgt_train_iter = iter(self.src_train), iter(self.tgt_train)
        optimizer = torch.optim.AdamW(
            [
                {"params": prepare.parameters(), "lr": 4e-4},
                {'params': universum.parameters(), "lr": 4e-4},
                {"params": encoder.parameters(), "lr": 1e-3},
                {"params": decoder.parameters(), "lr": 1e-3},
                # {"params": classifier.parameters(), "lr": 4e-4},
            ],
            betas=(0.93, 0.98),
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.3)
        src_correct_num = 0
        tgt_correct_num = 0
        src_total_num = 0
        tgt_total_num = 0
        L_Rec_v = []
        L_2_v = []
        L_u_v = []
        for step in range(self.iteration):
            try:
                src_batch = next(src_train_iter)
            except StopIteration:
                src_train_iter = iter(self.src_train)
                src_batch = next(src_train_iter)
            try:
                tgt_batch = next(tgt_train_iter)
            except StopIteration:
                tgt_train_iter = iter(self.tgt_train)
                tgt_batch = next(tgt_train_iter)
            src_x, src_y = self.get_data(src_batch)
            tgt_x, tgt_y = self.get_data(tgt_batch)
            src_batch_size = src_y.shape[0]
            tgt_batch_size = tgt_y.shape[0]
            x = torch.cat([src_x, tgt_x], dim=0)
            x = prepare(x)
            h_e = encoder(x)
            r_e = decoder(h_e)
            L_Rec = torch.nn.MSELoss()(x, r_e)
            L_Rec_v.append(L_Rec.data.item())
            h_u = universum(h_e)
            zero = torch.zeros([1, ]).to(device)
            L_2 = C * torch.square(torch.max(torch.abs(xi - torch.sum(torch.mul(h_u[:src_batch_size], src_y))), zero))

            L_2_v.append(L_2.data.item())
            U_h_u = torch.max(torch.abs(h_u[src_batch_size: src_batch_size + tgt_batch_size]) - epsilon, zero)

            L_u = torch.sum(U_h_u)
            L_u_v.append(L_u.data.item())

            y_ = classifier(h_e)
            L_cls = self.criterion(h_u[:src_batch_size], src_y)
            loss = L_Rec + C * L_2 + C_u * L_u
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            src_correct_num += accuracy_cal(h_u[:src_batch_size], src_y).cpu().numpy()
            src_total_num += src_batch_size
            tgt_correct_num += accuracy_cal(h_u[src_batch_size:src_batch_size + tgt_batch_size], tgt_y).cpu().numpy()
            tgt_total_num += tgt_batch_size
            if step % 50 == 0:
                scheduler.step()
                print(f"step: {step}")
                # print(torch.sum(torch.mul(h_u[:src_batch_size], src_y)))
                print(f"src acc: {src_correct_num / src_total_num}, \ttgt acc: {tgt_correct_num / tgt_total_num}")
                print(f"rec loss: {np.mean(L_Rec_v)}, \tL2 loss:{np.mean(L_2_v)}, \tLu loss: {np.mean(L_u_v)}")
                L_Rec_v = []
                L_2_v = []
                L_u_v = []
                src_correct_num = 0
                tgt_correct_num = 0
                src_total_num = 0
                tgt_total_num = 0

        torch.save(encoder, self.encoder_path)
        torch.save(universum, self.universum_path)

    def test(self):
        encoder = torch.load(self.encoder_path)
        classifier = torch.load(self.universum_path)
        model = nn.Sequential(
            encoder,
            classifier
        )
        correct_num = 0
        loss = 0
        for vx, vy in self.tgt_val:
            vx = vx.to(device)
            vy = vy.to(device)
            out = model(vx)
            correct_num += accuracy_cal(out, vy).cpu().numpy()
            loss += self.criterion(out, vy).data.item()
        print(f"test acc: {correct_num / len(self.tgt_val.dataset)}, "
              f"test loss: {loss / math.ceil(len(self.tgt_val.dataset) / self.batch_size)}")


if __name__ == "__main__":
    arg = Args()
    src_dataset_name = 'CASIA_'
    tgt_dataset_name = "RAVDESS_"
    trainer = UAETrainer(arg, src_dataset_name, tgt_dataset_name)
    trainer.train()
    # trainer.test()
