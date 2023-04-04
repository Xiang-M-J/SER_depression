import math
import os

import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from einops.layers.torch import Rearrange
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from CNN import CausalConv
from config import Args
from utils import load_multi_dataset, accuracy_cal, l2_regularization, get_newest_file, seed_everything

datasets = ['IEMOCAP', 'CASIA', 'RAVDESS']
num_class = [6, 6, 8]
seq_len = [313, 188, 188]
num_sample = [7200, 7200, 1440]  # IEMOCAP原本有7380个样本，随机丢弃了180个样本
split_rate = [0.8, 0.1, 0.1]
epochs = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_amp = True
if use_amp:
    scaler = GradScaler()


def train_step(model, optimizer, x, y, index, weight_decay):
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    if use_amp:
        with autocast():
            loss, correct_num = model(x, y, index)
            loss += l2_regularization(model, weight_decay)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss, correct_num = model(x, y, index)
        loss += l2_regularization(model, weight_decay)
        loss.backward()
        optimizer.step()
    return loss, correct_num


def val_step(model, x, y, index, weight_decay):
    x, y = x.to(device), y.to(device)
    if use_amp:
        with autocast():
            loss, correct_num = model(x, y, index)
            loss += l2_regularization(model, weight_decay)
    else:
        loss, correct_num = model(x, y, index)
        loss += l2_regularization(model, weight_decay)
    return loss, correct_num


def test_step(model, x, y, index, weight_decay):
    return val_step(model, x, y, index, weight_decay)


def train(arg: Args):
    source1_train_loader, source1_val_loader = load_multi_dataset(datasets[0], split_rate,
                                                                  arg.random_seed, arg.batch_size, out_type=True,
                                                                  length=7200)
    source2_train_loader, source2_val_loader = load_multi_dataset(datasets[1], split_rate,
                                                                  arg.random_seed, arg.batch_size, out_type=True,
                                                                  length=-1)
    source3_train_loader, source3_val_loader = load_multi_dataset(datasets[2], split_rate,
                                                                  arg.random_seed, arg.batch_size, out_type=True,
                                                                  length=-1)

    model = MultiDataset_3(arg)
    optimizer1 = torch.optim.Adam([{'params': model.shareNet.parameters(), 'lr': arg.lr * 0.1},
                                   {'params': model.classifier1.parameters(), 'lr': arg.lr}],
                                  betas=(arg.beta1, arg.beta2))  # 为每个数据集都分配一个优化器
    optimizer2 = torch.optim.Adam([{'params': model.shareNet.parameters(), 'lr': arg.lr * 0.1},
                                   {'params': model.classifier2.parameters(), 'lr': arg.lr}],
                                  betas=(arg.beta1, arg.beta2))
    optimizer3 = torch.optim.Adam([{'params': model.shareNet.parameters(), 'lr': arg.lr * 0.1},
                                   {'params': model.classifier3.parameters(), 'lr': arg.lr}],
                                  betas=(arg.beta1, arg.beta2))
    best_val_accuracy = 0
    model = model.to(device)
    train_batch = math.ceil(7200 * 0.8 / arg.batch_size)
    for epoch in range(epochs):
        source1_train_iter = iter(source1_train_loader)
        source2_train_iter = iter(source2_train_loader)
        source3_train_iter = iter(source3_train_loader)

        val_correct = [0, 0, 0]
        val_loss = [0, 0, 0]
        train_correct = [0, 0, 0]
        train_loss = [0, 0, 0]
        model.train()
        print("train...")
        for batch in tqdm(range(train_batch)):
            source1_tx, source1_ty = next(source1_train_iter)
            source2_tx, source2_ty = next(source2_train_iter)

            # dataset 1
            loss, correct_num = train_step(model, optimizer1, source1_tx, source1_ty,
                                           index=1, weight_decay=arg.weight_decay)
            train_correct[0] += correct_num
            train_loss[0] += loss.data.item()
            # dataset 2
            loss, correct_num = train_step(model, optimizer2, source2_tx, source2_ty,
                                           index=2, weight_decay=arg.weight_decay)
            train_correct[1] += correct_num
            train_loss[1] += loss.data.item()

            if batch % 5 == 0:  # IEMOCAP, CASIA, RAVDESS的比例为 5:5:1
                source3_tx, source3_ty = next(source3_train_iter)

                # dataset 3
                loss, correct_num = train_step(model, optimizer3, source3_tx, source3_ty,
                                               index=3, weight_decay=arg.weight_decay)
                train_correct[2] += correct_num
                train_loss[2] += loss.data.item()

        print(f"epoch {epoch}:")
        for i in range(3):
            train_acc = train_correct[i] / int(num_sample[i] * split_rate[0])
            print(f"{datasets[i]}: train Loss:{train_loss[i]:.4f}\t train Accuracy:{train_acc:.3f}\t")
        model.eval()
        print("eval...")
        with torch.no_grad():
            for s1_vx, s1_vy in source1_val_loader:
                loss, correct_num = val_step(model, s1_vx, s1_vy, 1, arg.weight_decay)
                val_correct[0] += correct_num
                val_loss[0] += loss.data.item()
            for s2_vx, s2_vy in source2_val_loader:
                loss, correct_num = val_step(model, s2_vx, s2_vy, 2, arg.weight_decay)
                val_correct[1] += correct_num
                val_loss[1] += loss.data.item()
            for s3_vx, s3_vy in source3_val_loader:
                loss, correct_num = val_step(model, s3_vx, s3_vy, 3, arg.weight_decay)
                val_correct[2] += correct_num
                val_loss[2] += loss.data.item()
        print(f"{epoch}:")
        for i in range(3):
            val_acc = val_correct[i] / int(num_sample[i] * split_rate[1])
            print(f"{datasets[i]}: val Loss:{val_loss[i]:.4f}\t val Accuracy:{val_acc:.3f}\t")

        torch.save(model, "models/pretrain_models/IEMOCAP_CASIA_RAVDESS.pt")
        # metric.train_acc.append(float(train_correct * 100) / train_num)
        # metric.train_loss.append(train_loss / math.ceil((train_num / batch_size)))
        # metric.val_acc.append(float(val_correct * 100) / val_num)
        # metric.val_loss.append(val_loss / math.ceil(val_num / batch_size))

        # print(
        #     'Epoch :{}\t train Loss:{:.4f}\t train Accuracy:{:.3f}\t val Loss:{:.4f} \t val Accuracy:{:.3f}'.format(
        #         epoch + 1, metric.train_loss[-1], metric.train_acc[-1], metric.val_loss[-1],
        #         metric.val_acc[-1]))
        # if metric.val_acc[-1] > best_val_accuracy:
        #     print(f"val_accuracy improved from {best_val_accuracy} to {metric.val_acc[-1]}")
        #     best_val_accuracy = metric.val_acc[-1]
        #     metric.best_val_acc[0] = best_val_accuracy
        #     metric.best_val_acc[1] = metric.train_acc[-1]
        #     if self.args.save:
        #         torch.save(model, self.best_path)
        #         print(f"saving model to {self.best_path}")
        # elif metric.val_acc[-1] == best_val_accuracy:
        #     if metric.train_acc[-1] > metric.best_val_acc[1]:
        #         metric.best_val_acc[1] = metric.train_acc[-1]
        #         if self.args.save:
        #             torch.save(model, self.best_path)
    #     else:
    #         print(f"val_accuracy did not improve from {best_val_accuracy}")
    # if self.args.save:
    #     torch.save(model, self.last_path)
    #     print(f"save model(last): {self.last_path}")
    #     plot(metric.item(), self.args.model_name, self.result_path)
    #     np.save(self.result_path + "data/" + self.args.model_name + "_train_metric.data", metric.item())
    #     self.writer.add_text("beat validation accuracy", f"{metric.best_val_acc}")
    #     dummy_input = torch.rand(self.args.batch_size, self.args.feature_dim, self.args.seq_len)
    #
    # return metric


def test(arg: Args, path=None):
    source1_test_loader = load_multi_dataset(datasets[0], split_rate, arg.random_seed, arg.batch_size, out_type=False,
                                             length=7200)
    source2_test_loader = load_multi_dataset(datasets[1], split_rate, arg.random_seed, arg.batch_size, out_type=False,
                                             length=-1)
    source3_test_loader = load_multi_dataset(datasets[2], split_rate, arg.random_seed, arg.batch_size, out_type=False,
                                             length=-1)

    if path is None:
        path = get_newest_file("models/pretrain_models/")
        print(f"path is None, choose the newest model: {path}")
    if not os.path.exists(path):
        print(f"error! cannot find the {path}")
        return
    model = torch.load(path)
    test_correct = [0, 0, 0]
    test_loss = [0, 0, 0]
    model.eval()
    print("test...")
    with torch.no_grad():
        for s1_tx, s1_ty in source1_test_loader:
            loss, correct_num = test_step(model, s1_tx, s1_ty, 1, arg.weight_decay)
            test_correct[0] += correct_num
            test_loss[0] += loss.data.item()
        for s2_tx, s2_ty in source2_test_loader:
            loss, correct_num = test_step(model, s2_tx, s2_ty, 2, arg.weight_decay)
            test_correct[1] += correct_num
            test_loss[1] += loss.data.item()
        for s3_tx, s3_ty in source3_test_loader:
            loss, correct_num = test_step(model, s3_tx, s3_ty, 3, arg.weight_decay)
            test_correct[2] += correct_num
            test_loss[2] += loss.data.item()
    for i in range(3):
        test_acc = test_correct[i] / int(num_sample[i] * split_rate[2])
        print(f"{datasets[i]}: val Loss:{test_loss[i]:.4f}\t val Accuracy:{test_acc:.3f}\t")


class MultiDataset_3(nn.Module):
    def __init__(self, arg: Args):
        super(MultiDataset_3, self).__init__()
        self.arg = Args
        self.shareNet = CausalConv(arg.feature_dim, arg.dilation, arg.filters, arg.kernel_size, arg.drop_rate)
        self.classifier1 = nn.Sequential(  # 用于IEMOCAP 6分类
            nn.AdaptiveAvgPool1d(1),
            Rearrange("N C L -> N (C L)"),
            nn.Dropout(arg.drop_rate),
            nn.Linear(in_features=arg.feature_dim, out_features=6),
        )
        self.classifier2 = nn.Sequential(  # 用于CASIA 6分类
            nn.AdaptiveAvgPool1d(1),
            Rearrange("N C L -> N (C L)"),
            nn.Dropout(arg.drop_rate),
            nn.Linear(in_features=arg.feature_dim, out_features=6),
        )
        self.classifier3 = nn.Sequential(  # 用于RAVDESS 8分类
            nn.AdaptiveAvgPool1d(1),
            Rearrange("N C L -> N (C L)"),
            nn.Dropout(arg.drop_rate),
            nn.Linear(in_features=arg.feature_dim, out_features=8),
        )

    def forward(self, x, y, index):
        x = self.shareNet(x)
        if index == 1:
            x = self.classifier1(x)
            loss = F.cross_entropy(x, y)
            correct_num = accuracy_cal(x, y)
            return loss, correct_num
        elif index == 2:
            x = self.classifier2(x)
            loss = F.cross_entropy(x, y)
            correct_num = accuracy_cal(x, y)
            return loss, correct_num
        elif index == 3:
            x = self.classifier3(x)
            loss = F.cross_entropy(x, y, label_smoothing=0.1)
            correct_num = accuracy_cal(x, y)
            return loss, correct_num


if __name__ == "__main__":
    arg = Args()
    seed_everything(arg.random_seed)
    train(arg)
