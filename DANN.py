import math

import numpy as np
import torch.optim

from blocks import CausalConv
import torch.nn as nn
from torch.autograd import Function
from einops.layers.torch import Rearrange
from config import Args
from utils import load_loader, accuracy_cal, Metric, dataset_num_class, cal_seq_len

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GRL(Function):
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x) * constant

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.constant, None


class DANNModel(nn.Module):
    def __init__(self, arg):
        super(DANNModel, self).__init__()
        self.feature_extractor = nn.Sequential(
            CausalConv(arg),
        )
        self.label_predictor = nn.Sequential(
            Rearrange("N C L -> N (C L)"),
            nn.Linear(arg.seq_len * arg.filters, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, arg.num_class[0]),
        )
        # self.target_label_predictor = None

        self.target_label_predictor = nn.Sequential(
            Rearrange("N C L -> N (C L)"),
            nn.Linear(arg.seq_len * arg.filters, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(100, arg.num_class[1]),
            # nn.Linear(arg.num_class[0], 50),
            # nn.Linear(50, arg.num_class[1])
        )
        self.domain_classifier = nn.Sequential(
            nn.Conv1d(in_channels=arg.filters, out_channels=arg.filters, kernel_size=3, padding="same"),
            nn.BatchNorm1d(arg.filters),
            nn.MaxPool1d(2),
            nn.ReLU(),
            nn.Conv1d(in_channels=arg.filters, out_channels=arg.filters, kernel_size=3, padding="same"),
            nn.BatchNorm1d(arg.filters),
            nn.MaxPool1d(2),
            nn.ReLU(),
            Rearrange("N C L -> N (C L)"),
            nn.Linear(cal_seq_len(arg.seq_len, 4) * arg.filters, 100),
            nn.Linear(100, 2),
            # nn.Linear(arg.seq_len * arg.filters, 100),
            # nn.BatchNorm1d(100),
            # nn.ReLU(),
            # nn.Linear(100, 2),
        )

    def forward(self, x, alpha, flag=False):
        """
        flag: true for target data, false for source data
        """
        x = self.feature_extractor(x)
        y = GRL.apply(x, alpha)

        # x = self.label_predictor(x)
        # if self.target_label_predictor is not None:
        #     x = self.target_label_predictor(x)
        # x = self.label_predictor(x)
        if flag:
            x = self.target_label_predictor(x)
        else:
            x = self.label_predictor(x)
        y = self.domain_classifier(y)
        return x, y

    def apply_grad(self, loss):
        grad = torch.autograd.grad(outputs=loss, inputs=self.target_label_predictor.parameters(),
                                   allow_unused=True)
        for p, g in zip(self.target_label_predictor.parameters(), grad):
            p.grad = g


class DANNTrainer:
    def __init__(self, arg, src_dataset_name, tgt_dataset_name):
        self.src_dt = src_dataset_name
        self.tgt_dt = tgt_dataset_name
        self.src_train, self.src_val, self.src_test = \
            load_loader(self.src_dt, arg.spilt_rate, arg.batch_size, arg.random_seed, arg.version, arg.order)
        self.tgt_train, self.tgt_val, self.tgt_test = \
            load_loader(self.tgt_dt, arg.spilt_rate, arg.batch_size, arg.random_seed)
        self.mini_seq_len = min(self.src_train.dataset.dataset.x.shape[-1], self.tgt_train.dataset.dataset.x.shape[-1])
        self.lr = arg.lr
        self.weight_decay = arg.weight_decay
        self.batch_size = arg.batch_size
        self.iteration = 2500
        arg.seq_len = self.mini_seq_len
        # arg.two_predictor = False
        arg.num_class = dataset_num_class([src_dataset_name, tgt_dataset_name])
        # if arg.num_class[0] != arg.num_class[1]:
        #     arg.two_predictor = True
        self.model = DANNModel(arg).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.model.feature_extractor.parameters(), 'lr': self.lr},
            {'params': self.model.label_predictor.parameters(), 'lr': self.lr},
            {'params': self.model.domain_classifier.parameters(), 'lr': self.lr}
        ])
        # if arg.two_predictor:
        self.optimizer_special = torch.optim.Adam(self.model.target_label_predictor.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.metric = Metric()

    def set_optimizer_lr(self, p):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr / (1. + 10 * p) ** 0.75
        for param_group in self.optimizer_special.param_groups:
            param_group['lr'] = self.lr / (1. + 10 * p) ** 0.75

    def train_step(self, src_batch, tgt_batch, p):

        src_x, src_y = src_batch
        src_x = src_x[:, :, :self.mini_seq_len]
        src_x, src_y = src_x.to(device), src_y.to(device)
        alpha = 2. / (1 + np.exp((-10. * p))) - 1
        self.set_optimizer_lr(p)
        batch_size = src_y.shape[0]
        # domain_label = torch.cat([torch.ones([batch_size, 1]), torch.zeros([batch_size, 1])], dim=1).float().to(device)
        domain_label = torch.zeros(batch_size).long().to(device)
        label, domain = self.model(src_x, alpha)
        correct_num_src = accuracy_cal(label, src_y)
        src_label_loss = self.criterion(label, src_y)
        src_domain_loss = self.criterion(domain, domain_label)
        # print(accuracy_cal(domain, torch.cat([torch.ones([batch_size, 1]), torch.zeros([batch_size, 1])], dim=1).float().to(device)))

        tgt_x, tgt_y = tgt_batch
        tgt_x = tgt_x[:, :, :self.mini_seq_len]
        tgt_x, tgt_y = tgt_x.to(device), tgt_y.to(device)
        batch_size = tgt_x.shape[0]
        # domain_label = torch.cat([torch.zeros([batch_size, 1]), torch.ones([batch_size, 1])], dim=1).float().to(device)
        domain_label = torch.ones(batch_size).long().to(device)
        label, domain = self.model(tgt_x, alpha, flag=True)
        correct_num_tgt = accuracy_cal(label, tgt_y)
        tgt_label_loss = self.criterion(label, tgt_y)   # 因为源域和目标域的数据分类不同，所以需要额外训练目标域的标签分类器
        tgt_domain_loss = self.criterion(domain, domain_label)
        # print(accuracy_cal(domain, torch.cat([torch.zeros([batch_size, 1]), torch.ones([batch_size, 1])], dim=1).float().to(device)))
        total_loss = src_label_loss + src_domain_loss + tgt_domain_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.optimizer_special.zero_grad()
        self.model.apply_grad(tgt_label_loss)
        self.optimizer_special.step()

        return correct_num_src.cpu().numpy(), correct_num_tgt.cpu().numpy(), \
            src_label_loss.data.item(), src_domain_loss.data.item(), tgt_domain_loss.data.item()

    def val_step(self, vx, vy, p):
        vx = vx[:, :, :self.mini_seq_len]
        vx, vy = vx.to(device), vy.to(device)
        alpha = 2. / (1 + np.exp((-10. * p))) - 1
        label, _ = self.model(vx, alpha, flag=True)
        correct_num = accuracy_cal(label, vy)
        loss = self.criterion(label, vy)
        return correct_num.cpu().numpy(), loss.data.item()

    def train(self):
        mini_iter = min(len(self.src_train), len(self.tgt_train))
        mini_sample_num = min(len(self.src_train.dataset), len(self.tgt_train.dataset))

        src_train_iter, tgt_train_iter = iter(self.src_train), iter(self.tgt_train)
        train_acc = [0, 0]
        train_loss = [0, 0, 0]
        best_eval_acc = 0
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
            self.model.train()
            p = step / self.iteration
            correct_num_src, correct_num_tgt, src_label_loss, src_domain_loss, tgt_domain_loss = \
                self.train_step(src_batch, tgt_batch, p)
            train_acc[0] += correct_num_src
            train_acc[1] += correct_num_tgt
            train_loss[0] += src_label_loss
            train_loss[1] += src_domain_loss
            train_loss[2] += tgt_domain_loss

            if step % mini_iter == 0:
                train_acc = [acc / mini_sample_num for acc in train_acc]
                train_loss = [loss / mini_iter for loss in train_loss]
                self.metric.train_acc.append(train_acc)
                self.metric.train_loss.append(train_loss)
                print(f"step {step}:")
                print(f"train acc(src): {train_acc[0]}\t train acc(tgt): {train_acc[1]}")
                print(f"source label loss: {train_loss[0]}\t source domain loss: {train_loss[1]}\t "
                      f"target domain loss: {train_loss[2]}")
                print("eval...")
                self.model.eval()
                val_acc = 0
                val_loss = 0
                with torch.no_grad():
                    for vx, vy in self.tgt_val:
                        correct_num, loss_v = self.val_step(vx, vy, p)
                        val_acc += correct_num
                        val_loss += loss_v
                val_acc /= len(self.tgt_val.dataset)
                val_loss /= math.ceil(len(self.tgt_val.dataset) / self.batch_size)
                self.metric.val_acc.append(val_acc)
                self.metric.val_loss.append(val_loss)
                print(f"val acc: {val_acc}, val loss: {val_loss}")
                if val_acc > best_eval_acc:
                    torch.save(self.model, "models/dann_iemocap_modma_best.pt")
                    print(f"val acc improved from {best_eval_acc} to {val_acc} ")
                    best_eval_acc = val_acc
                    self.metric.best_val_acc[0] = best_eval_acc
                    self.metric.best_val_acc[1] = train_acc[1]
                else:
                    print(f"val acc do not improve from {best_eval_acc}")
                train_acc = [0, 0]
                train_loss = [0, 0, 0]
        torch.save(self.model, "models/dann_iemocap_modma_final.pt")


if __name__ == "__main__":
    arg = Args()
    src_dataset_name = 'IEMOCAP'
    tgt_dataset_name = "MODMA"
    trainer = DANNTrainer(arg, src_dataset_name, tgt_dataset_name)
    trainer.train()
