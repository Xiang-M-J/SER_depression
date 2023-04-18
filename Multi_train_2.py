import math
import os

import numpy as np
from matplotlib import pyplot as plt
from torch.autograd import Function
import torch.optim
from torch.cuda.amp import autocast, GradScaler

from config import Args
from multiModel import MModel, Discriminator, mmd
from utils import get_newest_file, load_loader, NoamScheduler, Metric

dataset_name = ['MODMA', 'IEMOCAP']
num_class = [2, 6]
seq_len = [313, 313]
num_sample = [3549, 7766]
split_rate = [0.7, 0.2, 0.1]
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


class MultiTrainer:
    def __init__(self, args: Args):
        args.num_class = num_class
        # args.seq_len = seq_len
        self.optimizer_type = args.optimizer_type
        self.epochs = args.epochs
        self.iteration = 5000
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.best_path = "models/Multi/MODMA_best.pt"
        self.model_path = "models/Multi/MODMA.pt"
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
    def get_mmd_loss(model, train_batch):
        mmd_feature = []
        for i in range(dataset_num):
            x = model[i].get_mmd_feature(train_batch[i][0].to(device))
            mmd_feature.append(x)
        mini_shape = min([x.shape[0] for x in mmd_feature])
        mmd_loss = mmd(mmd_feature[0][:mini_shape].view(mini_shape, -1),
                       mmd_feature[1][:mini_shape].view(mini_shape, -1))
        return mmd_loss

    @staticmethod
    def get_domain_loss(model, discriminator, train_batch, p):
        generalFeature = []
        for i in range(dataset_num):
            x = model[i].get_generalFeature(train_batch[i][0].to(device))
            generalFeature.append(x)
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

    @staticmethod
    def get_data(batch):
        x, y = batch
        x, y = x.to(device), y.to(device)
        return x, y

    def train(self):
        mini_iter = min([len(self.loader[i][0]) for i in range(dataset_num)])
        # mini_sample_num = min(len(self.src1_train.dataset), len(self.src2_train.dataset), len(self.tgt_train.dataset))

        train_iter = [iter(self.loader[i][0]) for i in range(dataset_num)]
        model = []
        optimizer = []
        scheduler = []
        for i in range(dataset_num):
            model.append(MModel(arg, seq_len=seq_len[i], index=i).to(device))
            parameter = [
                {'params': model[i].prepare.parameters(), 'lr': self.lr},
                {'params': model[i].shareNet.parameters(), 'lr': self.lr},
                {'params': model[i].specialNet.parameters(), 'lr': self.lr}
            ]
            optimizer.append(
                self.get_optimizer(arg, parameter, lr=arg.lr)
            )
            scheduler.append(self.get_scheduler(optimizer[i], arg))
        parameter = []
        for i in range(dataset_num):
            parameter.append(
                {'params': model[i].specialNet.parameters(), 'lr': self.lr},
            )
        optimizer.append(
            self.get_optimizer(arg, parameter, lr=arg.lr)
        )
        scheduler.append(
            self.get_scheduler(optimizer[dataset_num], arg)
        )
        discriminator = Discriminator(arg).to(device)

        domain_parameters = []
        for i in range(dataset_num):
            domain_parameters.append(
                {'params': model[i].prepare.parameters(), 'lr': self.lr}
            )
            domain_parameters.append(
                {'params': model[i].shareNet.parameters(), 'lr': self.lr*0.5}
            )
        domain_parameters.append(
            {'params': discriminator.parameters(), 'lr': self.lr}
        )
        domain_optimizer = self.get_optimizer(arg, domain_parameters, lr=arg.lr)
        domain_scheduler = self.get_scheduler(domain_optimizer, arg)

        best_val_accuracy = 0
        metric = Metric()
        train_num = [0, 0, 0]
        domain_acc = 0
        domain_num = 0
        val_acc = [0, 0, 0]
        val_loss = [0, 0, 0]
        train_acc = [0, 0, 0]
        train_loss = [0, 0, 0]
        for step in range(self.iteration):
            train_batch = []
            for i in range(dataset_num):
                if ((step + 1) % 2 == 0) or (i == 0):
                    try:
                        batch = next(train_iter[i])
                    except StopIteration:
                        train_iter[i] = iter(self.loader[i][0])
                        batch = next(train_iter[i])
                    train_batch.append(batch)
            for i in range(1, dataset_num):
                model[i].shareNet.load_state_dict(model[0].shareNet.state_dict())

            if (step + 1) % 6 == 0:
                p = (step + 1) / self.iteration
                for i in range(dataset_num):
                    model[i].train()
                discriminator.train()
                domain_loss, correct_num = self.get_domain_loss(model, discriminator, train_batch, p)
                domain_optimizer.zero_grad()
                domain_loss.backward()
                domain_optimizer.step()
                domain_acc += correct_num.cpu().numpy()
                domain_num += len(train_batch[0][0])
                domain_num += len(train_batch[1][0])

                mmd_loss = self.get_mmd_loss(model, train_batch)
                optimizer[dataset_num].zero_grad()
                mmd_loss.backward()
                optimizer[dataset_num].step()

            for i in range(dataset_num):
                if ((step + 1) % 2 == 0) or (i == 0):
                    model[i].train()
                    if i > 0:
                        model[i].shareNet.load_state_dict(model[i - 1].shareNet.state_dict())
                    loss, correct_num = self.train_step(model[i], optimizer[i], train_batch[i])
                    train_acc[i] += correct_num.cpu().numpy()
                    train_loss[i] += loss.data.item()
                    train_num[i] += len(train_batch[i][0])

            # model[0].train()
            # loss, correct_num = self.train_step(model[0], optimizer[0], train_batch[0])

            # mini_shape = min(x_1.shape[0], x_2.shape[0])
            # mmd_loss = mmd(x_1[:mini_shape].view(mini_shape, -1), x_2[:mini_shape].view(mini_shape, -1))
            # loss = loss1 + loss2 + mmd_loss
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            if (step + 1) % mini_iter == 0:
                print(optimizer[0].param_groups[0]['lr'])
                for i in range(len(scheduler)):
                    scheduler[i].step()
                domain_scheduler.step()
                print(f"step {step}:")
                for i in range(dataset_num):
                    train_acc[i] = train_acc[i] / train_num[i]
                    train_loss[i] = train_loss[i] / mini_iter
                    metric.train_acc.append(train_acc[i])
                    metric.train_loss.append(train_loss[i])
                    print(
                        f"{dataset_name[i]}: train Loss:{train_loss[i]:.4f}\t train Accuracy:{train_acc[i] * 100:.3f}\t")
                domain_acc = domain_acc / domain_num
                print(f"domain accuracy: {domain_acc}")

                for i in range(dataset_num):
                    model[i].eval()
                    if i > 0:
                        model[i].shareNet.load_state_dict(model[i - 1].shareNet.state_dict())
                with torch.no_grad():
                    for i in range(dataset_num):
                        for batch in self.loader[i][1]:
                            loss, correct_num = self.val_step(model[i], batch)
                            val_acc[i] += correct_num.cpu().numpy()
                            val_loss[i] += loss.data.item()
                for i in range(dataset_num):
                    val_acc[i] = val_acc[i] / int(num_sample[i] * split_rate[1])
                    val_loss[i] = val_loss[i] / math.ceil(int(num_sample[i] * split_rate[1]) / self.batch_size)
                    metric.val_acc.append(val_acc[i])
                    metric.val_loss.append(val_loss[i])
                    print(f"{dataset_name[i]}: val Loss:{val_loss[i]:.4f}\t val Accuracy:{val_acc[i] * 100:.3f}\t")
                if val_acc[0] > best_val_accuracy:
                    best_val_accuracy = val_acc[0]
                    metric.best_val_acc[0] = train_acc[0]
                    metric.best_val_acc[1] = best_val_accuracy
                    torch.save(model[0], self.best_path)

                plt.clf()
                plt.figure(figsize=(6, 14))
                for i in range(dataset_num):
                    plt.subplot(dataset_num, 1, i + 1)
                    plt.plot(metric.train_acc[i::dataset_num])
                    plt.plot(metric.val_acc[i::dataset_num])
                    plt.ylabel("accuracy(%)")
                    plt.xlabel("epoch")
                    plt.legend([f'{dataset_name[i]} train acc', f'{dataset_name[i]} val acc'])
                    plt.title(f"{dataset_name[i]} train accuracy and validation accuracy")
                plt.pause(0.02)
                plt.ioff()  # 关闭画图的窗口
                train_num = [0, 0, 0]
                domain_acc = 0
                domain_num = 0
                val_acc = [0, 0, 0]
                val_loss = [0, 0, 0]
                train_acc = [0, 0, 0]
                train_loss = [0, 0, 0]
        np.save("results/data/Multi/MODMA.npy", metric.item())
        torch.save(model[0], self.model_path)
        # torch.save(model, "models/Multi/IEMOCAP_CASIA_MODMA.pt")
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
            path = get_newest_file("models/Multi/")
            print(f"path is None, choose the newest model: {path}")
        if not os.path.exists(path):
            print(f"error! cannot find the {path}")
            return
        model = torch.load(self.model_path)
        test_acc = [0, 0, 0]
        test_loss = [0, 0, 0]
        model.eval()
        print("test...")
        with torch.no_grad():
            for batch in self.loader[0][2]:
                loss, correct_num = self.test_step(model, batch)
                test_acc[0] += correct_num
                test_loss[0] += loss.data.item()
        print("final path")
        for i in range(dataset_num):
            test_num = num_sample[i] - int(num_sample[i] * split_rate[0]) - int(num_sample[i] * split_rate[1])
            test_acc[i] = test_acc[i] / test_num
            test_loss[i] = test_loss[i] / math.ceil(int(num_sample[i] * split_rate[2]) / self.batch_size)
            print(f"{dataset_name[i]}: test Loss:{test_loss[i]:.4f}\t test Accuracy:{test_acc[i] * 100:.3f}\t")

        model = torch.load(self.best_path)
        test_acc = [0, 0, 0]
        test_loss = [0, 0, 0]
        model.eval()
        print("test...")
        print("best path")
        with torch.no_grad():
            for batch in self.loader[0][2]:
                loss, correct_num = self.test_step(model, batch)
                test_acc[0] += correct_num
                test_loss[0] += loss.data.item()
        for i in range(dataset_num):
            test_num = num_sample[i] - int(num_sample[i] * split_rate[0]) - int(num_sample[i] * split_rate[1])
            test_acc[i] = test_acc[i] / test_num
            test_loss[i] = test_loss[i] / math.ceil(int(num_sample[i] * split_rate[2]) / self.batch_size)
            print(f"{dataset_name[i]}: test Loss:{test_loss[i]:.4f}\t test Accuracy:{test_acc[i] * 100:.3f}\t")


if __name__ == "__main__":
    arg = Args()
    trainer = MultiTrainer(arg)
    # print()
    trainer.train()
    trainer.test()

# preprocess/data/IEMOCAP_V1_order3.npy
