import math
# 存在一些问题
import numpy as np
import torch.nn as nn
import torch.optim
from einops.layers.torch import Rearrange
from torch.autograd import Function
from multiModel import MModel
from config import Args
from utils import load_loader, accuracy_cal, Metric, dataset_num_class, cal_seq_len

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_class = [2, 6]


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
        arg.num_class = num_class
        self.mmodel = MModel(arg, index=0)
        self.discriminator = nn.Sequential(
            nn.Conv1d(in_channels=arg.filters, out_channels=arg.filters, kernel_size=3, padding="same"),
            nn.BatchNorm1d(arg.filters),
            nn.ReLU(),
            nn.Conv1d(in_channels=arg.filters, out_channels=arg.filters, kernel_size=3, padding="same"),
            nn.BatchNorm1d(arg.filters),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(128),
            Rearrange("N C L -> N (C L)"),
            nn.Linear(128 * arg.filters, 100),
            nn.Dropout(0.3),
            nn.Linear(100, 2),
        )

    def forward(self, x, y, alpha):
        feature = self.mmodel.get_generalFeature(x)
        loss, correct_num = self.mmodel(x, y)
        feature = GRL.apply(feature, alpha)
        domain_label = self.discriminator(feature)
        return loss, correct_num, domain_label


class DANNTrainer:
    def __init__(self, arg, src_dataset_name, tgt_dataset_name):
        self.src_dt = src_dataset_name
        self.tgt_dt = tgt_dataset_name
        self.src_train, self.src_val, self.src_test = \
            load_loader(self.src_dt, arg.spilt_rate, arg.batch_size, arg.random_seed, "V1", arg.order)
        self.tgt_train, self.tgt_val, self.tgt_test = \
            load_loader(self.tgt_dt, arg.spilt_rate, arg.batch_size, arg.random_seed, "V1")
        self.mini_seq_len = min(self.src_train.dataset.dataset.x.shape[-1], self.tgt_train.dataset.dataset.x.shape[-1])
        self.lr = arg.lr
        self.weight_decay = arg.weight_decay
        self.batch_size = arg.batch_size
        self.iteration = 2500
        self.model = DANNModel(arg).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.metric = Metric()

    def set_optimizer_lr(self, p):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr / (1. + 10 * p) ** 0.75

    def train_step(self, src_batch, tgt_batch, p):
        src_x, src_y = src_batch
        src_x, src_y = src_x.to(device), src_y.to(device)
        tgt_x, tgt_y = tgt_batch
        tgt_x, tgt_y = tgt_x.to(device), tgt_y.to(device)
        alpha = 2. / (1 + np.exp((-10. * p))) - 1
        self.set_optimizer_lr(p)

        batch_size = src_x.shape[0]
        domain_label = torch.zeros(batch_size).long().to(device)
        src_label_loss, correct_num_src, domain = self.model(src_x, src_y, alpha)
        src_domain_loss = self.criterion(domain, domain_label)

        batch_size = tgt_x.shape[0]
        domain_label = torch.ones(batch_size).long().to(device)
        tgt_label_loss, correct_num_tgt, domain = self.model(tgt_x, tgt_y, alpha)
        tgt_domain_loss = self.criterion(domain, domain_label)

        total_loss = src_label_loss + src_domain_loss + tgt_domain_loss + tgt_label_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return correct_num_src.cpu().numpy(), correct_num_tgt.cpu().numpy(), \
            src_label_loss.data.item(), src_domain_loss.data.item(), tgt_domain_loss.data.item()

    def val_step(self, vx, vy, p):
        vx, vy = vx.to(device), vy.to(device)
        alpha = 2. / (1 + np.exp((-10. * p))) - 1
        loss, correct_num, _ = self.model(vx, vy, alpha)
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
        torch.save(self.model, "models/dann_daic_modma_final.pt")


if __name__ == "__main__":
    arg = Args()
    src_dataset_name = 'DAIC'
    tgt_dataset_name = "MODMA"
    trainer = DANNTrainer(arg, src_dataset_name, tgt_dataset_name)
    trainer.train()
