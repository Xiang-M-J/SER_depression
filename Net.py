import datetime
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tensorboardX import SummaryWriter
from torch.utils.data import dataloader

from model import CNN_Transformer, TIM, SET, SET_official, CNN_ML_Transformer, Transformer_TIM, MLTransformer_TIM, \
    Transformer, CNN_Transformer_error, AT_TIM
from utils import Metric, accuracy_cal, check_dir, MODMA_LABELS, plot_matrix, plot, logger, \
    l2_regularization, noam, IEMOCAP_LABELS, compare_key

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Net_Instance:
    def __init__(self, args):
        self.args = args
        self.model_type = args.model_type
        self.best_path = f"models/" + args.model_name + "_best" + ".pt"  # 模型保存路径(max val acc)
        self.last_path = f"models/" + args.model_name + ".pt"  # 模型保存路径(final)
        self.result_path = f"results/"  # 结果保存路径（分为数据和图片）
        check_dir()
        if args.save:
            self.logger = logger(self.args.model_name, self.args.addition())
            date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            self.writer = SummaryWriter("runs/" + date)

    def train(self, train_dataset, val_dataset, batch_size: int, epochs: int, lr: float, weight_decay: float,
              is_mask=False):

        if self.args.mode == "pretrain":
            # model = TIM(self.args)
            model = CNN_Transformer(self.args)
            # print(model.name)
        else:
            if self.model_type == "CNN_Transformer":
                model = CNN_Transformer(self.args)
            elif self.model_type == "TIM":
                model = TIM(self.args)
            elif self.model_type == "SET":
                model = SET(self.args)
            elif self.model_type == "SET_official":
                model = SET_official(self.args)
            elif self.model_type == "CNN_ML_Transformer":
                model = CNN_ML_Transformer(self.args)
            elif self.model_type == "Transformer_TIM":
                model = Transformer_TIM(self.args)
            elif self.model_type == "MLTransformer_TIM":
                model = MLTransformer_TIM(self.args)
            elif self.model_type == "Transformer":
                model = Transformer(self.args)
            elif self.model_type == "AT_TIM":
                model = AT_TIM(self.args)
            elif self.model_type == "error":
                model = CNN_Transformer_error(self.args)
            else:
                raise NotImplementedError

        if self.args.save:
            # wandb.init(
            #     # set the wandb project where this run will be logged
            #     project="SER_depression",
            #     config=self.args.item(),
            #     name=model.name + datetime.datetime.now().strftime("%H_%M_%S")
            # )
            self.logger.start()
            self.writer.add_text("model name", self.args.model_name)
            self.writer.add_text('addition', self.args.addition())
        metric = Metric()
        train_loader = dataloader.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
        )
        val_loader = dataloader.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
        )
        # train_num = len(train_loader.dataset)  # 当数据增强时这样能得到正确的训练集数量
        train_num = len(train_dataset)
        val_num = len(val_dataset)

        if self.args.load_weight:
            if self.args.mode != "train":
                print(f"only load pretrain weight in train mode, but now is {self.args.mode} mode")
                return
            else:
                model_dict = model.state_dict()
                pretrain_model = torch.load(self.args.pretrain_model_path)
                pretrain_model_dict = pretrain_model.state_dict().items()
                tgt_key = list(model_dict)[0]
                src_key = list(pretrain_model_dict)[0][0]
                src_key, tgt_key = compare_key(src_key, tgt_key)
                pretrained_dict = {k.replace(src_key, tgt_key): v
                                   for k, v in pretrain_model_dict
                                   if k.replace(src_key, tgt_key) in model_dict}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
        # fine_tune_lr_layers = list(map(id, model.generalFeatureExtractor.parameters()))
        # lr_layers = filter(lambda p: id(p) not in fine_tune_lr_layers, model.parameters())
        # for name, param in model.generalFeatureExtractor.named_parameters():
        #     param.requires_grad = False
        # optimizer = torch.optim.Adam(
        #     [
        #         # {'params': model.generalFeatureExtractor.parameters(), 'lr': 0.2*lr},
        #         {'params': lr_layers, 'lr': lr},
        #     ],
        #     betas=(self.args.beta1, self.args.beta2),
        # )
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, betas=(self.args.beta1, self.args.beta2))

        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)  # 相当于smooth_labels的功能
        if self.args.scheduler_type == 1:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.args.step_size, gamma=self.args.gamma, last_epoch=-1)

        if is_mask:
            train_indices = train_dataset.indices
            val_indices = val_dataset.indices
            masks = np.load("preprocess/mask.npy")
            train_masks = masks[train_indices]
            val_masks = masks[val_indices]

        best_val_accuracy = 0
        if torch.cuda.is_available():
            model = model.cuda()
            loss_fn = loss_fn.cuda()
        steps = 0  # 用于warmup
        plt.ion()
        for epoch in range(epochs):
            model.train()
            train_correct = 0
            train_loss = 0
            val_correct = 0
            val_loss = 0
            for step, (bx, by) in enumerate(train_loader):
                if self.args.scheduler_type == 2:
                    for p in optimizer.param_groups:
                        p['lr'] = self.args.initial_lr * noam(d_model=self.args.d_model,
                                                              step=steps + 1, warmup=self.args.warmup)
                if torch.cuda.is_available():
                    bx = bx.cuda()
                    by = by.cuda()
                if is_mask:
                    mask = train_masks[step * batch_size: (step * batch_size + bx.shape[0])]
                    output = model(bx, mask)
                else:
                    output = model(bx)
                loss = loss_fn(output, by) + l2_regularization(model, weight_decay)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_correct += accuracy_cal(output, by)
                train_loss += loss.data.item()
                steps += 1

            model.eval()
            with torch.no_grad():
                for step, (vx, vy) in enumerate(val_loader):
                    if torch.cuda.is_available():
                        vx = vx.cuda()
                        vy = vy.cuda()
                    if is_mask:
                        mask = val_masks[step * batch_size: (step * batch_size + vx.shape[0])]
                        output = model(vx, mask, train=False)
                    else:
                        output = model(vx)
                    loss = loss_fn(output, vy)
                    val_correct += accuracy_cal(output, vy)
                    val_loss += loss.data.item()
            if self.args.scheduler_type == 1:
                scheduler.step()

            metric.train_acc.append(float(train_correct * 100) / train_num)
            metric.train_loss.append(train_loss / math.ceil((train_num / batch_size)))
            metric.val_acc.append(float(val_correct * 100) / val_num)
            metric.val_loss.append(val_loss / math.ceil(val_num / batch_size))
            plt.clf()
            plt.plot(metric.train_acc)
            plt.plot(metric.val_acc)
            plt.legend(["train acc", "val acc"])
            plt.pause(0.02)
            plt.ioff()  # 关闭画图的窗口

            if self.args.save:
                self.writer.add_scalar('train accuracy', metric.train_acc[-1], epoch + 1)
                self.writer.add_scalar('train loss', metric.train_loss[-1], epoch + 1)
                self.writer.add_scalar('validation accuracy', metric.val_acc[-1], epoch + 1)
                self.writer.add_scalar('validation loss', metric.val_loss[-1], epoch + 1)
                # wandb.log({
                #     'epoch': epoch,
                #     'train_acc': metric.train_acc[-1],
                #     'train_loss': metric.train_loss[-1],
                #     'val_acc': metric.val_acc[-1],
                #     'val_loss': metric.val_loss[-1]
                # })
            print(
                'Epoch :{}\t train Loss:{:.4f}\t train Accuracy:{:.3f}\t val Loss:{:.4f} \t val Accuracy:{:.3f}'.format(
                    epoch + 1, metric.train_loss[-1], metric.train_acc[-1], metric.val_loss[-1],
                    metric.val_acc[-1]))
            if metric.val_acc[-1] > best_val_accuracy:
                print(f"val_accuracy improved from {best_val_accuracy} to {metric.val_acc[-1]}")
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
            if self.model_type == "TIM":
                self.writer.add_graph(model, dummy_input)
            else:
                self.writer.add_graph(model, [dummy_input, mask])
            self.logger.train(train_metric=metric)
        return metric

    def test(self, test_dataset, batch_size: int, model_path: str = None):
        if model_path is None:
            model_path = self.best_path
        if not os.path.exists(model_path):
            print(f"error! cannot find the model in {model_path}")
            return
        model = torch.load(model_path)
        print(f"load model: {model_path}")
        if self.args.save:
            self.logger.start()
        test_loader = dataloader.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
        )
        test_num = len(test_dataset)
        model.eval()
        metric = Metric(mode="test")
        test_correct = 0
        test_loss = 0
        loss_fn = torch.nn.CrossEntropyLoss()
        y_pred = torch.zeros(test_num)
        y_true = torch.zeros(test_num)
        if torch.cuda.is_available():
            loss_fn = loss_fn.cuda()
        for step, (vx, vy) in enumerate(test_loader):
            if torch.cuda.is_available():
                vx = vx.cuda()
                vy = vy.cuda()
            with torch.no_grad():
                output = model(vx)
                y_pred[step * batch_size: step * batch_size + vy.shape[0]] = torch.max(output.data, 1)[1]
                y_true[step * batch_size: step * batch_size + vy.shape[0]] = torch.max(vy.data, 1)[1]
                loss = loss_fn(output, vy)
                test_correct += accuracy_cal(output, vy)
                test_loss += loss
        conf_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(self.args.num_class))
        metric.confusion_matrix = conf_matrix
        if self.args.mode == "pretrain":
            fig = plot_matrix(conf_matrix, labels_name=IEMOCAP_LABELS, model_name=self.args.model_name, normalize=False,
                              result_path=self.result_path)
            report = classification_report(y_true, y_pred, target_names=IEMOCAP_LABELS)
        else:
            fig = plot_matrix(conf_matrix, labels_name=MODMA_LABELS, model_name=self.args.model_name, normalize=False,
                              result_path=self.result_path)
            report = classification_report(y_true, y_pred, target_names=MODMA_LABELS)
        print(report)
        metric.report = report
        metric.test_loss = test_loss / math.ceil(test_num / batch_size)
        metric.test_acc = float((test_correct * 100) / test_num)
        print("{} test Loss:{:.4f} test Accuracy:{:.3f}".format(
            self.args.model_name, metric.test_loss, metric.test_acc))
        if self.args.save:
            self.writer.add_text("test loss", str(metric.test_loss))
            self.writer.add_text("test accuracy", str(metric.test_acc))
            self.writer.add_figure("confusion matrix", fig)
            self.writer.add_text("classification report", report)
            self.logger.test(test_metric=metric)
            np.save(self.result_path + "data/" + self.args.model_name + "_test_metric.npy", metric.item())
        return metric
