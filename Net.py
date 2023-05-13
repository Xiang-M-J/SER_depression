import datetime
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tensorboardX import SummaryWriter
from torch.utils.data import dataloader

from baseline import TIM, SET, Transformer_TIM, Transformer, LSTM, TemporalConvNet, MLP
from config import Args
from model import AT_TIM, Transformer_DeltaTIM, AT_DeltaTIM, MTCN
from utils import Metric, accuracy_cal, check_dir, MODMA_LABELS, plot_matrix, plot, logger, EarlyStopping, \
    IEMOCAP_LABELS, NoamScheduler, ModelSave, EarlyStoppingLoss

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Agent:
    """
    训练智能体
    """
    def __init__(self, args: Args):
        self.args: Args = args
        self.optimizer_type = args.optimizer_type
        self.model_type = args.model_type
        self.best_path = f"models/" + args.model_name + "_best" + ".pt"  # 模型保存路径(max val acc)
        self.last_path = f"models/" + args.model_name + ".pt"  # 模型保存路径(final)
        self.result_path = f"results/"  # 结果保存路径（分为数据和图片）
        self.save_path = f"models/" + args.model_name
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.lr = args.lr
        self.test_acc = []
        check_dir()
        if args.save:
            self.logger = logger(self.args.model_name, self.args.addition())
            date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            self.writer = SummaryWriter("runs/" + date)

    def get_optimizer(self, parameter, lr):
        if self.optimizer_type == 0:
            optimizer = torch.optim.SGD(params=parameter, lr=lr, weight_decay=self.args.weight_decay)
            # 对于SGD而言，L2正则化与weight_decay等价
        elif self.optimizer_type == 1:
            optimizer = torch.optim.Adam(params=parameter, lr=lr, betas=(self.args.beta1, self.args.beta2),
                                         weight_decay=self.args.weight_decay)
            # 对于Adam而言，L2正则化与weight_decay不等价
        elif self.optimizer_type == 2:
            optimizer = torch.optim.AdamW(params=parameter, lr=lr, betas=(self.args.beta1, self.args.beta2),
                                          weight_decay=self.args.weight_decay)
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

    def train(self, train_dataset, val_dataset, test_dataset):
        if self.model_type == "TIM":
            model = TIM(self.args)
        elif self.model_type == "SET":
            model = SET(self.args)
        elif self.model_type == "Transformer_TIM":
            model = Transformer_TIM(self.args)
        elif self.model_type == "Transformer":
            model = Transformer(self.args)
        elif self.model_type == "AT_TIM":
            model = AT_TIM(self.args)
        elif self.model_type == "Transformer_DeltaTIM":
            model = Transformer_DeltaTIM(self.args)
        elif self.model_type == "AT_DeltaTIM":
            model = AT_DeltaTIM(self.args)
        elif self.model_type == "MTCN":
            model = MTCN(self.args)
        elif self.model_type == "MLP":
            model = MLP(self.args)
        elif self.model_type == "LSTM":
            model = LSTM(self.args)
        elif self.model_type == "TCN":
            model = TemporalConvNet(self.args)
        else:
            raise NotImplementedError

        if self.args.save:
            self.logger.start()
            self.writer.add_text("model name", self.args.model_name)
            self.writer.add_text('addition', self.args.addition())
        metric = Metric()
        train_loader = dataloader.DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
        )
        val_loader = dataloader.DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
        )
        # train_num = len(train_loader.dataset)  # 当数据增强时这样能得到正确的训练集数量
        train_num = len(train_dataset)
        val_num = len(val_dataset)

        if self.args.load_weight:
            model_dict = model.state_dict()
            pretrain_model = torch.load(self.args.pretrain_model_path)
            print(f"load: {self.args.pretrain_model_path}")
            pretrain_model_dict = pretrain_model.state_dict().items()
            # 对于Transformer_DeltaTIM
            # pretrain_name = ['prepare', 'generalFeatureExtractor']
            #
            # for k, v in pretrain_model_dict:
            #     if k.split('.')[0] in pretrain_name:
            #         model_dict.update({k: v})
            # model.load_state_dict(model_dict)
            # parameter = []
            # for name, param in model.named_parameters():
            #     if name.split('.')[0] in pretrain_name:
            #         # param.requires_grad = False
            #         parameter.append({'params': param, 'lr': 0.2 * lr})
            #     else:
            #         parameter.append({"params": param, "lr": lr})
            # optimizer = self.get_optimizer(parameter, lr)

            # 对于AT_DeltaTIM
            # pretrain_name = ['dilation_layer']
            #
            # for k, v in pretrain_model_dict:
            #     if k.split('.')[0] in pretrain_name:
            #         # param.requires_grad = False
            #         if int(k.split('.')[1]) < 4:
            #             model_dict.update({k: v})
            #
            # model.load_state_dict(model_dict)
            # parameter = []
            # for name, param in model.named_parameters():
            #     if name.split('.')[0] in pretrain_name:
            #         # param.requires_grad = False
            #         if int(name.split('.')[1]) < 4:
            #             parameter.append({'params': param, 'lr': 0.2 * lr})
            #         else:
            #             parameter.append({'params': param, 'lr': lr})
            #     else:
            #         parameter.append({"params": param, "lr": lr})
            # optimizer = self.get_optimizer(parameter, lr)

            # 对于MTCN
            # pretrain_name = ['generalExtractor', 'specialExtractor']
            pretrain_name = ['generalExtractor']  # 只固定 generalExtractor的效果更好一点
            # for k, v in pretrain_model_dict:
            #     if k.split('.')[0] != "classifier":
            #         model_dict.update({k: v})
            model.block.block1.load_state_dict(pretrain_model.block.block1.state_dict())
            parameter = []
            for name, param in model.named_parameters():
                if name.split('.')[0] in pretrain_name:
                    parameter.append({"params": param, "lr": 0.2 * self.lr})
                    # param.requires_grad = False
                else:
                    parameter.append({"params": param, "lr": self.lr})
            optimizer = self.get_optimizer(parameter, self.lr)
            # optimizer = torch.optim.AdamW([
            #     # {"params": model.attn.parameters(), 'lr': self.args.lr},
            #     {'params': model.classifier.parameters(), 'lr': self.args.lr},
            # ])
            # 其它
            # tgt_key = list(model_dict)[0]
            # src_key = list(pretrain_model_dict)[0][0]
            # src_key, tgt_key = compare_key(src_key, tgt_key)
            # pretrained_dict = {k.replace(src_key, tgt_key): v
            #                    for k, v in pretrain_model_dict
            #                    if k.replace(src_key, tgt_key) in model_dict}
            # model_dict.update(pretrained_dict)
        else:
            optimizer = self.get_optimizer(model.parameters(), self.lr)

        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1).to(device)  # label_smoothing=0.1 相当于smooth_labels的功能
        early_stop = EarlyStopping(patience=self.args.patience)
        # early_stop = EarlyStoppingLoss(patience=5, delta_loss=2e-4)
        # model_save = ModelSave(save_path=self.save_path, )
        scheduler = self.get_scheduler(optimizer, arg=self.args)
        best_val_accuracy = 0
        model = model.to(device)
        steps = 0  # 用于warmup
        plt.ion()
        for epoch in range(self.epochs):
            model.train()
            train_correct = 0
            train_loss = 0
            val_correct = 0
            val_loss = 0
            for step, (bx, by) in enumerate(train_loader):
                bx, by = bx.to(device), by.to(device)
                output = model(bx)
                loss = loss_fn(output, by)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_correct += accuracy_cal(output, by)
                train_loss += loss.data.item()
                steps += 1

            model.eval()
            with torch.no_grad():
                for step, (vx, vy) in enumerate(val_loader):
                    vx, vy = vx.to(device), vy.to(device)
                    output = model(vx)
                    loss = loss_fn(output, vy)
                    val_correct += accuracy_cal(output, vy)
                    val_loss += loss.data.item()

            if self.args.scheduler_type == 0:
                pass
            elif self.args.scheduler_type == 2:
                scheduler.step(steps)
            else:
                scheduler.step()

            metric.train_acc.append(float(train_correct * 100) / train_num)
            metric.train_loss.append(train_loss / math.ceil((train_num / self.batch_size)))
            metric.val_acc.append(float(val_correct * 100) / val_num)
            metric.val_loss.append(val_loss / math.ceil(val_num / self.batch_size))

            plt.clf()
            plt.plot(metric.train_acc)
            plt.plot(metric.val_acc)
            plt.ylabel("accuracy(%)")
            plt.xlabel("epoch")
            plt.legend(["train acc", "val acc"])
            plt.title("train accuracy and validation accuracy")
            plt.pause(0.02)
            plt.ioff()  # 关闭画图的窗口

            if self.args.save:
                self.writer.add_scalar('train accuracy', metric.train_acc[-1], epoch + 1)
                self.writer.add_scalar('train loss', metric.train_loss[-1], epoch + 1)
                self.writer.add_scalar('validation accuracy', metric.val_acc[-1], epoch + 1)
                self.writer.add_scalar('validation loss', metric.val_loss[-1], epoch + 1)

            print(
                'Epoch :{}\t train Loss:{:.4f}\t train Accuracy:{:.3f}\t val Loss:{:.4f} \t val Accuracy:{:.3f}'.format(
                    epoch + 1, metric.train_loss[-1], metric.train_acc[-1], metric.val_loss[-1],
                    metric.val_acc[-1]))
            print(optimizer.param_groups[0]['lr'])
            if metric.val_acc[-1] > best_val_accuracy:
                print(f"val_accuracy improved from {best_val_accuracy :.3f} to {metric.val_acc[-1]:.3f}")
                best_val_accuracy = metric.val_acc[-1]
                metric.best_val_acc[0] = best_val_accuracy
                metric.best_val_acc[1] = metric.train_acc[-1]
                if self.args.save:
                    torch.save(model, self.best_path)
                    print(f"saving model to {self.best_path}")
            elif metric.val_acc[-1] == best_val_accuracy:
                if metric.train_acc[-1] > metric.best_val_acc[1]:
                    metric.best_val_acc[1] = metric.train_acc[-1]
                    if self.args.save:
                        torch.save(model, self.best_path)
            else:
                print(f"val_accuracy did not improve from {best_val_accuracy}")
            if early_stop(metric.val_acc[-1]):
                break
            # if early_stop(metric.val_loss[-1]):
            #     break
            if metric.val_acc[-1] > 99.4:
                self.test_acc.append(self.multi_test_step(model, test_dataset=test_dataset))

            # model_save(model, epoch)

        if self.args.save:
            torch.save(model, self.last_path)
            print(f"save model(last): {self.last_path}")
            plot(metric.item(), self.args.model_name, self.result_path)
            np.save(self.result_path + "data/" + self.args.model_name + "_train_metric", metric.item())
            self.writer.add_text("beat validation accuracy", f"{metric.best_val_acc}")
            self.writer.add_text("parameter setting", self.args.addition())
            self.writer.add_text("model name", model.name)

            dummy_input = torch.rand(self.args.batch_size, self.args.feature_dim, self.args.seq_len).to(device)
            mask = torch.rand(self.args.seq_len, self.args.seq_len).to(device)
            if self.model_type in ["TIM", "LSTM", "TCN"]:
                self.writer.add_graph(model, dummy_input)
            else:
                self.writer.add_graph(model, [dummy_input, mask])
            self.logger.train(train_metric=metric)

    def test_step(self, model_path, test_loader, loss_fn, test_num, metric, best=False):
        """
        Args:
            model_path: 模型路径

        Returns:
            metric fig
        """
        if not os.path.exists(model_path):
            print(f"error! cannot find the model in {model_path}")
            return
        print(f"load model: {model_path}")
        model = torch.load(model_path)
        model.eval()
        test_correct = 0
        test_loss = 0
        y_pred = torch.zeros(test_num)
        y_true = torch.zeros(test_num)
        for step, (vx, vy) in enumerate(test_loader):
            vx, vy = vx.to(device), vy.to(device)
            with torch.no_grad():
                output = model(vx)
                y_pred[step * self.batch_size: step * self.batch_size + vy.shape[0]] = torch.max(output.data, 1)[1]
                y_true[step * self.batch_size: step * self.batch_size + vy.shape[0]] = torch.max(vy.data, 1)[1]
                loss = loss_fn(output, vy)
                test_correct += accuracy_cal(output, vy)
                test_loss += loss.data.item()
        conf_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(self.args.num_class))

        fig = plot_matrix(conf_matrix, labels_name=MODMA_LABELS, model_name=self.args.model_name, normalize=False,
                          result_path=self.result_path, best=best)
        try:
            report = classification_report(y_true, y_pred, target_names=MODMA_LABELS)
        except ValueError:
            report = classification_report(y_true, y_pred, target_names=IEMOCAP_LABELS)
        test_acc = float((test_correct * 100) / test_num)
        test_loss = test_loss / math.ceil(test_num / self.batch_size)
        metric.confusion_matrix.append(conf_matrix)
        metric.report.append(report)
        metric.test_loss.append(test_loss)
        metric.test_acc.append(test_acc)
        return metric, fig

    def test(self, test_dataset, model_path: str = None):
        metric = Metric(mode="test")
        metric.report = []
        metric.confusion_matrix = []
        metric.test_acc = []
        metric.test_loss = []
        if self.args.save:
            self.logger.start()
        test_loader = dataloader.DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
        )
        loss_fn = torch.nn.CrossEntropyLoss().to(device)
        test_num = len(test_dataset)
        if model_path is None:
            model_path = self.last_path
            figs = []
            metric, fig = self.test_step(model_path, test_loader, loss_fn, test_num, metric)
            print(metric.report[0])

            figs.append(fig)
            print("{} final test Loss:{:.4f} test Accuracy:{:.3f}".format(
                self.args.model_name, metric.test_loss[0], metric.test_acc[0]))

            model_path = self.best_path
            metric, fig = self.test_step(model_path, test_loader, loss_fn, test_num, metric, True)
            print(metric.report[1])
            figs.append(fig)
            print("{} best test Loss:{:.4f} test Accuracy:{:.3f}".format(
                self.args.model_name, metric.test_loss[1], metric.test_acc[1]))

            if self.args.save:
                self.writer.add_text("test loss(final)", str(metric.test_loss[0]))
                self.writer.add_text("test accuracy(final)", str(metric.test_acc[0]))
                self.writer.add_figure("confusion matrix(final)", figs[0])
                self.writer.add_text("classification report(final)", metric.report[0])
                self.writer.add_text("test loss(best)", str(metric.test_loss[1]))
                self.writer.add_text("test accuracy(best)", str(metric.test_acc[1]))
                self.writer.add_figure("confusion matrix(best)", figs[1])
                self.writer.add_text("classification report(best)", metric.report[1])
                self.logger.test(test_metric=metric)
                np.save(self.result_path + "data/" + self.args.model_name + "_test_metric.npy", metric.item())
        else:
            metric, fig = self.test_step(model_path, test_loader, loss_fn, test_num, metric)
            print("{} test Loss:{:.4f} test Accuracy:{:.3f}".format(
                self.args.model_name, metric.test_loss[0], metric.test_acc[0]))

            if self.args.save:
                self.writer.add_text("test loss(final)", str(metric.test_loss[0]))
                self.writer.add_text("test accuracy(final)", str(metric.test_acc[0]))
                self.writer.add_figure("confusion matrix(final)", fig)
                self.writer.add_text("classification report(final)", metric.report[0])
                self.logger.test(test_metric=metric)
                np.save(self.result_path + "data/" + self.args.model_name + "_test_metric.npy", metric.item())

    def multi_test(self, test_dataset):
        """
        测试每隔step个epoch保存的模型
        """
        models = os.listdir(self.save_path)
        metric = Metric(mode="test")
        metric.report = []
        metric.confusion_matrix = []
        metric.test_acc = []
        metric.test_loss = []
        test_loader = dataloader.DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
        )
        loss_fn = torch.nn.CrossEntropyLoss().to(device)
        test_num = len(test_dataset)
        for m in models:
            model_path = self.save_path + f"/{m}"
            metric, _ = self.test_step(model_path, test_loader, loss_fn, test_num, metric)
        print(metric.test_acc)

    def multi_test_step(self, model, test_dataset):
        test_loader = dataloader.DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
        )
        test_num = len(test_dataset)
        test_acc = 0
        model.eval()
        with torch.no_grad():
            for vx, vy in test_loader:
                vx, vy = vx.to(device), vy.to(device)
                with torch.no_grad():
                    output = model(vx)
                    test_acc += accuracy_cal(output, vy).cpu().numpy()
        test_acc = test_acc / test_num
        return test_acc
