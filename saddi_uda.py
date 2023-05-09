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
    def __init__(self, in_dim, out_dim, scale_factor, kernel_size, size=None):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding="same")
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
        self.conv5 = ConvBlock(256, 512, 3, 2)
        self.final = Rearrange("N C L -> N (C L)")

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.final(x)
        return x


class Generator(nn.Module):
    def __init__(self, arg: Args, is_train=False):
        super(Generator, self).__init__()
        self.upsample1 = UpsampleBlock(514, 256, 3, 2) if not is_train else UpsampleBlock(512, 256, 3, 2)
        self.upsample2 = UpsampleBlock(256, 128, 3, 4)
        self.upsample3 = UpsampleBlock(128, 64, 3, 4)
        self.upsample4 = UpsampleBlock(64, 64, 3, 4)
        self.upsample5 = UpsampleBlock(64, arg.feature_dim, 2, 15, size=313)

    def forward(self, x):
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)
        x = self.upsample5(x)
        return x


class Classifier(nn.Module):
    def __init__(self, arg: Args):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 2),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, arg, is_train=False):
        super(Discriminator, self).__init__()
        self.is_train = is_train
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=arg.filters, out_channels=arg.filters, kernel_size=3, padding="same"),
            nn.BatchNorm1d(arg.filters),
            # nn.MaxPool1d(2),
            nn.ReLU(),
            Rearrange("N C L -> N (C L)"),
            nn.Linear(arg.seq_len * arg.filters, 100),
            nn.Linear(100, 3) if not is_train else nn.Linear(100, 1),  # 3 = 2(情绪类别) + 1
        )

    def forward(self, x):
        x = self.net(x)
        if self.is_train:
            x = nn.Sigmoid()(x)
        return x


class sADDiTrainer:
    def __init__(self, arg: Args, src_dataset_name, tgt_dataset_name):
        self.src_dt = src_dataset_name
        self.tgt_dt = tgt_dataset_name
        self.src_train, self.src_val = load_loader(self.src_dt, [0.8, 0.2], arg.batch_size, arg.random_seed, "V1")
        self.tgt_train, self.tgt_val = load_loader(self.tgt_dt, [0.8, 0.2], arg.batch_size, arg.random_seed, "V1")
        self.lr = arg.lr
        self.weight_decay = arg.weight_decay
        self.batch_size = arg.batch_size
        self.feature_dim = arg.feature_dim
        self.pretext_epochs = 50
        self.iteration = 3000
        self.encoder_final_path = f"models/uda/encoder_final.pt"
        self.encoder_train_path = f"models/uda/encoder_train.pt"
        self.classifier_path = "models/uda/classifier.pt"
        self.pretrain_best_path = f"models/uda/pretrain_best.pt"
        self.pretrain_final_path = f"models/uda/pretrain_final.pt"
        self.tgt_best_path = f"models/uda/tgt_encoder_best.pt"
        self.tgt_final_path = f"models/uda/tgt_encoder_final.pt"
        self.critic_best_path = f"models/uda/critic_best.pt"
        self.critic_final_path = f"models/uda/critic_final.pt"
        self.arg = arg
        self.seq_len = 313
        self.AECriterion = nn.MSELoss()
        self.GANCriterion = nn.BCELoss()
        self.criterion = nn.CrossEntropyLoss()
        self.metric = Metric()

    @staticmethod
    def pseudo_label(batch_size):
        label = torch.zeros([batch_size, 2])
        num = int(batch_size / 2)
        label[:num, 0] = 1.
        label[num:, 1] = 1.
        return label[torch.randperm(label.size(0))].type(torch.FloatTensor)

    @staticmethod
    def get_data(batch):
        x, y = batch
        x, y = x.to(device), y.to(device)
        return x, y

    def pretext(self):
        encoder = Encoder(self.arg).to(device)
        generator = Generator(self.arg).to(device)

        discriminator = Discriminator(self.arg).to(device)
        ae_optimizer = torch.optim.RMSprop(
            [{"params": encoder.parameters(), "lr": 6e-4},
             {"params": generator.parameters(), "lr": 6e-4}], weight_decay=0.2
        )
        discriminator_optimizer = torch.optim.RMSprop(
            [{"params": discriminator.parameters(), "lr": 6e-4}], weight_decay=0.2
        )
        generator_optimizer = torch.optim.RMSprop(
            [{"params": generator.parameters(), "lr": 6e-4}], weight_decay=0.2
        )
        # src_train_iter = iter(self.src_train)

        for epoch in range(self.pretext_epochs):
            # 记录变量置0
            ae_loss = []
            disc_loss = []
            gene_loss = []
            disc_num = 0
            disc_acc = 0
            gene_num = 0
            gene_acc = 0
            for real_x, real_label in self.src_train:
                batch_size = real_x.shape[0]
                real_x, real_label = real_x.to(device), real_label.to(device)

                # 训练Discriminator
                encoder.eval()
                generator.eval()
                discriminator.train()
                x_no = torch.randn([batch_size, self.feature_dim, self.seq_len]).to(device)  # 不带情绪的数据
                z_p = encoder(x_no)
                y_p = self.pseudo_label(x_no.shape[0]).to(device)  # 伪标签

                label = torch.cat([torch.argmax(real_label, dim=1), torch.full([batch_size], 2).to(device)], dim=0)
                fake_x = generator(torch.cat([z_p, y_p], dim=1).unsqueeze(-1))
                X = torch.cat([real_x, fake_x], dim=0)
                # label = torch.cat([fake_label, real_label], dim=0)
                Y = discriminator(X)
                epoch_disc_loss = self.criterion(Y, label)
                correct_num = accuracy_cal(Y, label)
                disc_acc += correct_num.cpu().numpy()
                disc_num += Y.shape[0]
                discriminator_optimizer.zero_grad()
                epoch_disc_loss.backward()
                discriminator_optimizer.step()
                disc_loss.append(epoch_disc_loss.data.item())

                # 训练AE
                encoder.train()
                generator.train()
                discriminator.eval()
                x_no = torch.randn([batch_size, self.feature_dim, self.seq_len]).to(device)  # 不带情绪的数据
                z_p = encoder(x_no)
                y_p = self.pseudo_label(x_no.shape[0]).to(device)  # 伪标签
                fake_x = generator(torch.cat([z_p, y_p], dim=1).unsqueeze(-1))
                epoch_ae_loss = self.AECriterion(x_no, fake_x)
                ae_optimizer.zero_grad()
                epoch_ae_loss.backward()
                ae_optimizer.step()
                ae_loss.append(epoch_ae_loss.data.item())

                # 训练generator
                encoder.eval()
                generator.train()
                discriminator.eval()
                x_no = torch.randn([batch_size, self.feature_dim, self.seq_len]).to(device)
                z_p = encoder(x_no)
                y_p = self.pseudo_label(x_no.shape[0]).to(device)  # 伪标签
                label = torch.argmax(y_p, dim=1).to(device)
                fake_x = generator(torch.cat([z_p, y_p], dim=1).unsqueeze(-1))
                fake_y = discriminator(fake_x)
                epoch_gene_loss = self.criterion(fake_y, label)
                correct_num = accuracy_cal(fake_y, label)
                gene_num += fake_y.shape[0]
                gene_acc += correct_num.cpu().numpy()
                generator_optimizer.zero_grad()
                epoch_gene_loss.backward()
                generator_optimizer.step()
                gene_loss.append(epoch_gene_loss.data.item())
            print(f"epoch{epoch}: {np.mean(ae_loss)},  {np.mean(disc_loss)},  {np.mean(gene_loss)}")
            print(f"discriminator acc {disc_acc / disc_num}, generator acc {gene_acc / gene_num}")
        torch.save(encoder, self.encoder_final_path)

    def train(self, flag=True):
        encoder = Encoder(self.arg).to(device)
        if os.path.exists(self.encoder_final_path) and flag:
            pretrain_encoder = torch.load(self.encoder_final_path)
            encoder.load_state_dict(pretrain_encoder.state_dict())
        src_batch_size = int(self.batch_size * 0.8)
        tgt_batch_size = self.batch_size - src_batch_size
        self.src_train = DataLoader(self.src_train.dataset, batch_size=src_batch_size, shuffle=False)
        self.tgt_train = DataLoader(self.tgt_train.dataset, batch_size=tgt_batch_size, shuffle=False)
        src_train_iter, tgt_train_iter = iter(self.src_train), iter(self.tgt_train)

        generator = Generator(self.arg, is_train=True).to(device)
        discriminator_src = Discriminator(self.arg, is_train=True).to(device)
        discriminator_tgt = Discriminator(self.arg, is_train=True).to(device)
        classifier = Classifier(self.arg).to(device)
        optimizer = torch.optim.AdamW(
            [{'params': discriminator_src.parameters(), "lr": 3e-4},
             {"params": discriminator_tgt.parameters(), "lr": 3e-4},
             {'params': classifier.parameters(), "lr": 3e-4},
             {"params": encoder.parameters(), "lr": 6e-4},
             {"params": generator.parameters(), "lr": 6e-4}], weight_decay=0.2
        )

        correct_num = 0
        total_num = 0
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
            X = torch.cat([src_x, tgt_x], dim=0)
            # Y = torch.cat([src_y, tgt_y], dim=0)
            z_d = encoder(X)
            emotion = classifier(z_d[:src_batch_size])
            class_loss = self.criterion(emotion, src_y)
            correct_num += accuracy_cal(emotion, src_y)
            total_num += src_batch_size
            domain_code = torch.cat([torch.zeros(src_batch_size), torch.ones(tgt_batch_size)], dim=0)
            r = generator(z_d.unsqueeze(-1))
            ae_loss = self.AECriterion(r, X)
            src_disc_input = torch.cat([src_x, r[src_batch_size:]], dim=0)
            src_disc_label = torch.cat([torch.ones([src_batch_size, 1]),
                                        torch.zeros([tgt_batch_size, 1])], dim=0).to(device)
            src_disc_output = discriminator_src(src_disc_input)
            src_disc_loss = self.GANCriterion(src_disc_output, src_disc_label)
            tgt_disc_input = torch.cat([tgt_x, r[:src_batch_size]], dim=0)
            tgt_disc_label = torch.cat([torch.ones([tgt_batch_size, 1]),
                                        torch.zeros([src_batch_size, 1])], dim=0).to(device)
            tgt_disc_output = discriminator_tgt(tgt_disc_input)
            tgt_disc_loss = self.GANCriterion(tgt_disc_output, tgt_disc_label)

            total_loss = class_loss + src_disc_loss + tgt_disc_loss + ae_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if (step + 1) % 100 == 0:
                accuracy = correct_num / total_num
                correct_num = 0
                total_num = 0
                print(f"step{step + 1}: {accuracy}")
        torch.save(encoder, self.encoder_train_path)
        torch.save(classifier, self.classifier_path)

    def test(self):
        encoder = torch.load(self.encoder_train_path)
        classifier = torch.load(self.classifier_path)
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
    src_dataset_name = 'DAIC21'
    tgt_dataset_name = "MODMA"
    trainer = sADDiTrainer(arg, src_dataset_name, tgt_dataset_name)
    trainer.pretext()
    trainer.train()
    trainer.test()
