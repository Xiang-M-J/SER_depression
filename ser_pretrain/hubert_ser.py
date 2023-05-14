import math

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from transformers import HubertPreTrainedModel, HubertModel, AutoConfig

from data_module import PretrainDataModule
from pretrain_utils import train_step, val_step
from utils import Metric

model_name_or_path = "TencentGameMate/chinese-hubert-base"
num_class = 2
epochs = 30
pooling_mode = 'mean'
spilt_rate = [0.6, 0.2, 0.2]
use_amp = True
label_list = ['HC', 'MDD']
batch_size = 16
lr = 2e-4
beta1 = 0.93
beta2 = 0.98

config = AutoConfig.from_pretrained(
    pretrained_model_name_or_path=model_name_or_path,
)
setattr(config, 'num_class', num_class)
setattr(config, 'pooling_mode', pooling_mode)


class HubertClassificationHead(nn.Module):
    """Head for hubert classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self.dropout = nn.Dropout(config.final_dropout)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(config.hidden_size, config.num_class)

    def forward(self, x, **kwargs):
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def merged_strategy(hidden_states, mode="mean"):
    if mode == "mean":
        outputs = torch.mean(hidden_states, dim=1)
    elif mode == "sum":
        outputs = torch.sum(hidden_states, dim=1)
    elif mode == "max":
        outputs = torch.max(hidden_states, dim=1)[0]
    else:
        raise Exception(
            "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")
    return outputs


class HubertForSpeechClassification(HubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_class = config.num_class
        self.pooling_mode = config.pooling_mode
        self.config = config
        self.hubert = HubertModel(self.config)
        self.classifier = HubertClassificationHead(config)

        self.init_weights()

    def forward(
            self,
            input_values,
            # attention_mask=None,
            # output_attentions=None,
            # output_hidden_states=None,
    ):
        outputs = self.hubert(
            input_values,
            # attention_mask=attention_mask,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        return logits


def train(dataset, paths: list):
    train_num = dataset.train_num
    val_num = dataset.val_num

    train_loader = dataset.train_dataloader(batch_size=batch_size)
    val_loader = dataset.val_dataloader(batch_size=batch_size)

    model = HubertForSpeechClassification.from_pretrained(
        model_name_or_path,
        config=config,
    )
    model.gradient_checkpointing_enable()
    parameter = []
    for name, param in model.named_parameters():  # 仅训练classifier.dense和classifier.out_proj
        if "hubert" in name:
            # param.requires_grad = False
            parameter.append({'params': param, 'lr': 0.2 * lr})
        else:
            parameter.append({'params': param, "lr": lr})

    optimizer = torch.optim.AdamW(parameter, lr=lr, betas=(beta1, beta2), weight_decay=0.2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)
    if use_amp:
        scaler = GradScaler()
    else:
        scaler = None

    if torch.cuda.is_available():
        model = model.cuda()
    metric = Metric()
    best_val_accuracy = 0
    for epoch in range(epochs):
        model.train()
        train_correct = 0
        train_loss = 0
        val_correct = 0
        val_loss = 0
        for bx, by in tqdm(train_loader):
            correct_num, loss_v = train_step(model, optimizer, bx, by, use_amp, scaler)
            train_correct += correct_num.cpu().numpy()
            train_loss += loss_v
        model.eval()
        with torch.no_grad():
            for vx, vy in tqdm(val_loader):
                correct_num, loss_v = val_step(model, vx, vy, use_amp)
                val_correct += correct_num.cpu().numpy()
                val_loss += loss_v
        train_acc = train_correct / train_num
        train_loss = train_loss / (math.ceil(train_num / batch_size))
        val_acc = val_correct / val_num
        val_loss = val_loss / (math.ceil(train_num / batch_size))
        metric.train_acc.append(train_acc*100)
        metric.train_loss.append(train_loss)
        metric.val_acc.append(val_acc*100)
        metric.val_loss.append(val_loss)
        print(
            "epoch: {}, train_accuracy: {:.3f}\t train_loss: {:.4f}; \t val_accuracy: {:.3f}\t val_loss: {:.4f}".format(
                epoch + 1, train_acc, train_loss, val_acc, val_loss))
        if val_acc >= best_val_accuracy:
            torch.save(model, paths[1])
            best_val_accuracy = val_acc
            metric.best_val_acc[0] = best_val_accuracy
            metric.best_val_acc[1] = train_acc
            print(f"best val accuracy: {best_val_accuracy}")
        scheduler.step()
        np.save(paths[2], metric.item())
        torch.save(model, paths[0])
        state = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), 'epoch': epoch}
        torch.save(state, paths[3])


def m_test(model_path:str, dataset):
    model = torch.load(model_path)
    test_acc = 0
    test_loss = []
    test_num = dataset.test_num
    test_loader = dataset.test_dataloader(batch_size=batch_size)
    with torch.no_grad():
        for vx, vy in tqdm(test_loader):
            correct_num, loss_v = val_step(model, vx, vy, use_amp)
            test_acc += correct_num.cpu().numpy()
            test_loss.append(loss_v)
    test_acc = test_acc / test_num
    test_loss = np.mean(test_loss)
    print(model_path)
    print("test accuracy: ", test_acc)
    print("test loss: ", test_loss)


if __name__ == "__main__":
    paths = ['models/hubert_ser_modma_final.pt', 'models/hubert_ser_modma_best.pt',
             "results/hubert_modma.npy", "results/hubert_optim_modma.pth"]
    dataset = PretrainDataModule(model_name_or_path, 2, "y/modma_y.npy", "x/MODMA/", duration=10)
    dataset.setup()
    train(dataset, paths)
    m_test(paths[0], dataset)
    m_test(paths[1], dataset)
