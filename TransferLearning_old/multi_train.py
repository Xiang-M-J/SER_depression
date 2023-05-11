import math
import os

import torch.optim
from torch.cuda.amp import autocast, GradScaler

from config import Args
from multiModel import Multi_model
from utils import get_newest_file, load_loader, NoamScheduler

dataset_name = ['IEMOCAP', 'CASIA', 'MODMA']
num_class = [6, 6, 2]
seq_len = [313, 188, 313]
num_sample = [7766, 7200, 3549]
split_rate = [0.7, 0.2, 0.1]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_amp = False
if use_amp:
    scaler = GradScaler()


class MultiTrainer:
    def __init__(self, args: Args):
        args.num_class = num_class
        # args.seq_len = seq_len
        self.optimizer_type = args.optimizer_type
        self.epochs = args.epochs
        self.iteration = 3000
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        args.spilt_rate = split_rate
        self.src1_train, self.src1_val, self.src1_test = \
            load_loader(dataset_name[0], split_rate, args.batch_size, args.random_seed, version='V1', order=3)
        self.src2_train, self.src2_val, self.src2_test = \
            load_loader(dataset_name[1], split_rate, args.batch_size, args.random_seed, version='V1', order=3)
        self.tgt_train, self.tgt_val, self.tgt_test = \
            load_loader(dataset_name[2], split_rate, args.batch_size, args.random_seed, version='V1', order=3)

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

    def train_step(self, model, optimizer, batch, index):
        x, y = self.get_data(batch)
        optimizer.zero_grad()
        if use_amp:
            with autocast():
                loss, correct_num = model(x, y, index)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, correct_num = model(x, y, index)
            loss.backward()
            optimizer.step()
        return loss, correct_num

    def val_step(self, model, batch, index):
        x, y = self.get_data(batch)
        if use_amp:
            with autocast():
                loss, correct_num = model(x, y, index)
        else:
            loss, correct_num = model(x, y, index)
        return loss, correct_num

    def test_step(self, model, batch, index):
        return self.val_step(model, batch, index)

    @staticmethod
    def get_data(batch):
        x, y = batch
        x, y = x.to(device), y.to(device)
        return x, y

    def train(self):
        mini_iter = min(len(self.src1_train), len(self.src2_train), len(self.tgt_train))
        # mini_sample_num = min(len(self.src1_train.dataset), len(self.src2_train.dataset), len(self.tgt_train.dataset))

        src1_train_iter, src2_train_iter, tgt_train_iter = \
            iter(self.src1_train), iter(self.src2_train), iter(self.tgt_train)

        model = Multi_model(arg, seq_len=seq_len)
        # 为每个数据集都分配一个优化器
        optimizer1 = torch.optim.AdamW([
                                        {'params': model.shareNet.parameters(), 'lr': self.lr * 0.6},
                                        {'params': model.specialNet1.parameters(), 'lr': self.lr}],
                                       betas=(arg.beta1, arg.beta2), weight_decay=self.weight_decay)
        optimizer2 = torch.optim.AdamW([
                                        {'params': model.shareNet.parameters(), 'lr': self.lr * 0.6},
                                        {'params': model.specialNet2.parameters(), 'lr': self.lr}],
                                       betas=(arg.beta1, arg.beta2), weight_decay=self.weight_decay)
        optimizer3 = torch.optim.AdamW([
            {'params': model.shareNet.parameters(), 'lr': self.lr},
                                        {'params': model.specialNet3.parameters(), 'lr': self.lr}],
                                       betas=(arg.beta1, arg.beta2), weight_decay=self.weight_decay)
        scheduler1 = self.get_scheduler(optimizer1, arg)
        scheduler2 = self.get_scheduler(optimizer2, arg)
        scheduler3 = self.get_scheduler(optimizer3, arg)
        # optimizer = torch.optim.AdamW(params=model.parameters(), lr=self.lr, betas=(arg.beta1, arg.beta2),
        #                               weight_decay=self.weight_decay)
        # scheduler = self.get_scheduler(optimizer, arg)
        best_val_accuracy = 0
        model = model.to(device)
        # train_batch = math.ceil(7200 * 0.8 / arg.batch_size)
        train_num = [0, 0, 0]
        val_acc = [0, 0, 0]
        val_loss = [0, 0, 0]
        train_acc = [0, 0, 0]
        train_loss = [0, 0, 0]
        for step in range(self.iteration):
            try:
                src1_batch = next(src1_train_iter)
            except StopIteration:
                src1_train_iter = iter(self.src1_train)
                src1_batch = next(src1_train_iter)
            try:
                src2_batch = next(src2_train_iter)
            except StopIteration:
                src2_train_iter = iter(self.src2_train)
                src2_batch = next(src2_train_iter)

            model.train()
            # dataset 1
            loss1, correct_num = self.train_step(model, optimizer1, src1_batch, index=0)
            train_acc[0] += correct_num
            train_loss[0] += loss1.data.item()
            train_num[0] += len(src1_batch[0])
            # dataset 2
            loss2, correct_num = self.train_step(model, optimizer2, src2_batch, index=1)
            train_acc[1] += correct_num
            train_loss[1] += loss2.data.item()
            train_num[1] += len(src2_batch[0])

            # if (step + 1) % 2 == 0:
            try:
                tgt_batch = next(tgt_train_iter)
            except StopIteration:
                tgt_train_iter = iter(self.tgt_train)
                tgt_batch = next(tgt_train_iter)

            # dataset 3
            loss3, correct_num = self.train_step(model, optimizer3, tgt_batch, index=2)
            train_acc[2] += correct_num
            train_loss[2] += loss3.data.item()
            train_num[2] += len(tgt_batch[0])

            # loss = (loss1 + loss2 + loss3)/3
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            if (step + 1) % mini_iter == 0:
                scheduler1.step()
                scheduler2.step()
                scheduler3.step()
                # scheduler.step()
                print(f"step {step}:")
                for i in range(3):
                    train_acc[i] = train_acc[i] / train_num[i]
                    train_loss[i] = train_loss[i] / mini_iter
                    print(f"{dataset_name[i]}: train Loss:{train_loss[i]:.4f}\t train Accuracy:{train_acc[i]:.3f}\t")

                model.eval()
                with torch.no_grad():
                    for batch in self.src1_val:
                        loss, correct_num = self.val_step(model, batch, 0)
                        val_acc[0] += correct_num
                        val_loss[0] += loss.data.item()
                    for batch in self.src2_val:
                        loss, correct_num = self.val_step(model, batch, 1)
                        val_acc[1] += correct_num
                        val_loss[1] += loss.data.item()
                    for batch in self.tgt_val:
                        loss, correct_num = self.val_step(model, batch, 2)
                        val_acc[2] += correct_num
                        val_loss[2] += loss.data.item()
                for i in range(3):
                    val_acc[i] = val_acc[i] / int(num_sample[i] * split_rate[1])
                    val_loss[i] = val_loss[i] / math.ceil(int(num_sample[i] * split_rate[1]) / self.batch_size)
                    print(f"{dataset_name[i]}: val Loss:{val_loss[i]:.4f}\t val Accuracy:{val_acc[i]:.3f}\t")

                train_num = [0, 0, 0]
                val_acc = [0, 0, 0]
                val_loss = [0, 0, 0]
                train_acc = [0, 0, 0]
                train_loss = [0, 0, 0]

        torch.save(model, "models/Multi/IEMOCAP_CASIA_MODMA.pt")
        # metric.train_acc.append(float(train_acc * 100) / train_num)
        # metric.train_loss.append(train_loss / math.ceil((train_num / batch_size)))
        # metric.val_acc.append(float(val_acc * 100) / val_num)
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

    def test(self, path=None):
        if path is None:
            path = get_newest_file("../models/Multi/")
            print(f"path is None, choose the newest model: {path}")
        if not os.path.exists(path):
            print(f"error! cannot find the {path}")
            return
        model = torch.load(path)
        test_acc = [0, 0, 0]
        test_loss = [0, 0, 0]
        model.eval()
        print("test...")
        with torch.no_grad():
            for batch in self.src1_test:
                loss, correct_num = self.test_step(model, batch, 1)
                test_acc[0] += correct_num
                test_loss[0] += loss.data.item()
            for batch in self.src2_test:
                loss, correct_num = self.test_step(model, batch, 2)
                test_acc[1] += correct_num
                test_loss[1] += loss.data.item()
            for batch in self.tgt_test:
                loss, correct_num = self.test_step(model, batch, 3)
                test_acc[2] += correct_num
                test_loss[2] += loss.data.item()
        for i in range(3):
            test_acc[i] = test_acc[i] / int(num_sample[i] * split_rate[2])
            test_loss[i] = test_loss[i] / math.ceil(int(num_sample[i] * split_rate[2]) / self.batch_size)
            print(f"{dataset_name[i]}: val Loss:{test_loss[i]:.4f}\t val Accuracy:{test_acc[i]:.3f}\t")


if __name__ == "__main__":
    arg = Args()
    trainer = MultiTrainer(arg)
    # print()
    trainer.train()
    # trainer.test()

# preprocess/data/IEMOCAP_V1_order3.npy
