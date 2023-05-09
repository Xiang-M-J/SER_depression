import math

import numpy as np
import torch.nn as nn
import torch.optim
from einops.layers.torch import Rearrange

from config import Args
from model import AT_DIFF_Block
from utils import load_loader, accuracy_cal, Metric

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Encoder(nn.Module):
    def __init__(self, arg: Args):
        super(Encoder, self).__init__()
        self.net = AT_DIFF_Block(arg)

    def forward(self, x):
        x = self.net(x)
        return x


class Classifier(nn.Module):
    def __init__(self, arg):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("N C L -> N (C L)"),
            nn.Linear(arg.filters, 2),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, arg):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=arg.filters, out_channels=arg.filters, kernel_size=3, padding="same"),
            nn.BatchNorm1d(arg.filters),
            # nn.MaxPool1d(2),
            nn.ReLU(),
            nn.Conv1d(in_channels=arg.filters, out_channels=arg.filters, kernel_size=3, padding="same"),
            nn.BatchNorm1d(arg.filters),
            # nn.MaxPool1d(2),
            nn.ReLU(),
            nn.Conv1d(in_channels=arg.filters, out_channels=arg.filters, kernel_size=3, padding="same"),
            nn.BatchNorm1d(arg.filters),
            # nn.MaxPool1d(2),
            nn.ReLU(),
            Rearrange("N C L -> N (C L)"),
            nn.Linear(arg.seq_len * arg.filters, 100),
            nn.Linear(100, 2),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class ADDATrainer:
    def __init__(self, arg, src_dataset_name, tgt_dataset_name):
        self.src_dt = src_dataset_name
        self.tgt_dt = tgt_dataset_name
        self.src_train, self.src_val = load_loader(self.src_dt, [0.8, 0.2], arg.batch_size, arg.random_seed)
        self.tgt_train, self.tgt_val, self.tgt_test = \
            load_loader(self.tgt_dt, [0.6, 0.2, 0.2], arg.batch_size, arg.random_seed)
        self.lr = arg.lr
        self.weight_decay = arg.weight_decay
        self.batch_size = arg.batch_size
        self.iteration = 2000
        self.pretrain_epochs = 15
        self.pretrain_best_path = f"models/uda/pretrain_DAIC21_best.pt"
        self.pretrain_final_path = f"models/uda/pretrain_DAIC21_final.pt"
        self.tgt_best_path = f"models/uda/tgt_encoder_best.pt"
        self.tgt_final_path = f"models/uda/tgt_encoder_final.pt"
        self.critic_best_path = f"models/uda/critic_best.pt"
        self.critic_final_path = f"models/uda/critic_final.pt"
        self.arg = arg
        self.criterion = nn.CrossEntropyLoss()
        self.metric = Metric()

    def pretrain(self):
        print("pretrain...")
        model = nn.Sequential(
            Encoder(self.arg),
            Classifier(self.arg)
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=4e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
        src_train_num = len(self.src_train.dataset)
        best_val_accuracy = 0
        metric = Metric()
        for epoch in range(self.pretrain_epochs):
            model.train()
            train_acc = 0
            train_loss = 0
            for bx, by in self.src_train:
                correct_num, loss_v = self.train_step(model, optimizer, bx, by)
                train_acc += correct_num
                train_loss += loss_v
            train_acc /= src_train_num
            train_loss /= math.ceil(src_train_num / self.batch_size)
            metric.train_acc.append(train_acc)
            metric.train_loss.append(train_loss)
            print(f"epoch {epoch + 1}: train_acc: {train_acc*100 :.3f}\t train_loss: {train_loss:.4f}")

            model.eval()
            val_acc = 0
            val_loss = 0
            src_val_num = len(self.src_val.dataset)
            with torch.no_grad():
                for vx, vy in self.src_val:
                    correct_num, loss_v = self.val_step(model, vx, vy)
                    val_acc += correct_num
                    val_loss += loss_v
            val_acc /= src_val_num
            val_loss /= math.ceil(src_val_num / self.batch_size)
            metric.val_acc.append(val_acc)
            metric.val_loss.append(val_loss)
            print(f"epoch {epoch + 1}: val_acc: {val_acc*100 :.3f}\t val_loss: {val_loss:.4f}")
            # scheduler.step()

            if val_acc > best_val_accuracy:
                print(f"val_accuracy improved from {best_val_accuracy*100:.3f} to {val_acc*100:.3f}")
                best_val_accuracy = val_acc
                metric.best_val_acc = [best_val_accuracy, train_acc]
                torch.save(model, self.pretrain_best_path)
                print(f"saving model to {self.pretrain_best_path}")
            elif val_acc == best_val_accuracy:
                if train_acc > metric.best_val_acc[1]:
                    metric.best_val_acc[1] = train_acc
                    print("update train accuracy")
            else:
                print(f"val_accuracy did not improve from {best_val_accuracy*100:.3f}")
        torch.save(model, self.pretrain_final_path)

    def train_step(self, model, optimizer, src_x, src_y):
        src_x, src_y = src_x.to(device), src_y.to(device)
        label = model(src_x)
        correct_num = accuracy_cal(label, src_y)
        loss = self.criterion(label, src_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return correct_num.cpu().numpy(), loss.data.item()

    def val_step(self, model, vx, vy):
        vx, vy = vx.to(device), vy.to(device)
        label = model(vx)
        correct_num = accuracy_cal(label, vy)
        loss = self.criterion(label, vy)
        return correct_num.cpu().numpy(), loss.data.item()

    @staticmethod
    def get_data(batch):
        x, y = batch
        x, y = x.to(device), y.to(device)
        return x, y

    def train(self):
        mini_iter = min(len(self.src_train), len(self.tgt_train))
        # 加载预训练模型
        tmp_model = torch.load(self.pretrain_final_path).to(device)
        src_encoder = tmp_model[0]

        # 初始化目标域编码器
        tgt_encoder = Encoder(self.arg).to(device)
        tgt_encoder.load_state_dict(src_encoder.state_dict())
        optimizer_tgt = torch.optim.Adam(tgt_encoder.parameters(), lr=1e-4)

        # 初始化判别器
        critic = Discriminator(arg).to(device)
        optimizer_critic = torch.optim.Adam(critic.parameters(), lr=1e-4)

        src_train_iter, tgt_train_iter = iter(self.src_train), iter(self.tgt_train)

        # 记录变量置0
        best_domain_tgt_acc = 0
        critic_domain_acc = 0
        critic_domain_loss = 0
        tgt_domain_acc = 0
        tgt_domain_loss = 0
        critic_sample_num = 0
        critic_update_iter = 0
        tgt_sample_num = 0
        metric = Metric()
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

            tgt_encoder.train()
            critic.train()
            src_x, src_y = self.get_data(src_batch)
            tgt_x, tgt_y = self.get_data(tgt_batch)

            src_batch_size = src_y.shape[0]
            tgt_batch_size = tgt_y.shape[0]
            tgt_sample_num += tgt_batch_size

            if step % 5 == 0:
                # 每5步训练一次discriminator
                src_label = torch.ones(src_batch_size).long().to(device)
                tgt_label = torch.zeros(tgt_batch_size).long().to(device)
                concat_label = torch.cat([src_label, tgt_label], 0)
                src_feature = src_encoder(src_x)
                tgt_feature = tgt_encoder(tgt_x)
                concat_feature = torch.cat([src_feature, tgt_feature], 0)
                concat_pred = critic(concat_feature.detach())
                critic_loss = self.criterion(concat_pred, concat_label)

                optimizer_critic.zero_grad()
                critic_loss.backward()
                optimizer_critic.step()

                correct_num = accuracy_cal(concat_pred, concat_label)
                critic_domain_acc += correct_num.cpu().numpy()
                critic_domain_loss += critic_loss.data.item()
                critic_sample_num += (src_batch_size + tgt_batch_size)
                critic_update_iter += 1

            # 训练目标域编码器
            tgt_feature = tgt_encoder(tgt_x)
            tgt_pred = critic(tgt_feature)
            tgt_label = torch.ones(tgt_batch_size).long().to(device)
            tgt_loss = self.criterion(tgt_pred, tgt_label)

            optimizer_tgt.zero_grad()
            optimizer_critic.zero_grad()
            tgt_loss.backward()
            optimizer_tgt.step()

            tgt_domain_acc += accuracy_cal(tgt_pred, tgt_label).cpu().numpy()
            tgt_domain_loss += tgt_loss.data.item()

            if (step + 1) % mini_iter == 0:
                critic_domain_acc /= critic_sample_num
                critic_domain_loss /= critic_update_iter
                tgt_domain_acc /= tgt_sample_num
                tgt_domain_loss /= mini_iter

                print(f"step {step}:")
                print(f"domain acc(critic): {critic_domain_acc * 100 :.3f}\t domain acc(tgt): {tgt_domain_acc * 100 :.3f}")
                print(f"domain loss(critic): {critic_domain_loss :.4f}\t domain loss(tgt): {tgt_domain_loss :.4f}")
                metric.train_acc.append(critic_domain_acc)
                metric.val_acc.append(tgt_domain_acc)
                metric.train_loss.append(critic_domain_loss)
                metric.val_loss.append(tgt_domain_loss)

                if tgt_domain_acc > best_domain_tgt_acc:

                    torch.save(tgt_encoder, self.tgt_best_path)
                    torch.save(critic, self.critic_best_path)
                    print(f"tgt domain acc improved from {best_domain_tgt_acc*100 :.3f} to {tgt_domain_acc*100 :.3f} ")
                    best_domain_tgt_acc = tgt_domain_acc
                    metric.best_val_acc[0] = best_domain_tgt_acc
                    metric.best_val_acc[1] = tgt_domain_acc
                else:
                    print(f"tgt domain acc do not improve from {best_domain_tgt_acc*100 :.3f}")

                # 记录变量置零
                critic_domain_acc = 0
                critic_domain_loss = 0
                tgt_domain_acc = 0
                tgt_domain_loss = 0
                critic_sample_num = 0
                critic_update_iter = 0
                tgt_sample_num = 0
        np.save(f"results/data/uda/train.npy", metric.item())
        torch.save(tgt_encoder, self.tgt_final_path)
        torch.save(critic, self.critic_final_path)

    def test(self):
        tgt_encoder = torch.load(self.tgt_best_path)
        pretrain_model = torch.load(self.pretrain_best_path)
        model = nn.Sequential(
            tgt_encoder,
            pretrain_model[1],
        )
        metric = Metric(mode="test")
        test_acc = 0
        test_loss = 0
        with torch.no_grad():
            for bx, by in self.tgt_val:
                correct_num, loss_v = self.val_step(model, bx, by)
                test_acc += correct_num
                test_loss += loss_v
        test_acc /= len(self.tgt_val.dataset)
        test_loss /= math.ceil(len(self.tgt_val.dataset) / self.batch_size)
        metric.test_acc = test_acc
        metric.test_loss = test_loss
        print(f"test acc: {test_acc}, test loss: {test_loss}")
        np.save(f"results/data/uda/test.npy", metric.item())


if __name__ == "__main__":
    # 准确率：0.667
    arg = Args()
    src_dataset_name = 'DAIC21'
    tgt_dataset_name = "MODMA"
    trainer = ADDATrainer(arg, src_dataset_name, tgt_dataset_name)
    # trainer.pretrain()
    # trainer.test()
    trainer.train()
    trainer.test()
