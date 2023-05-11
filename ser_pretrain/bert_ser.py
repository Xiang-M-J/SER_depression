import math

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from transformers import AutoConfig, HubertConfig
from transformers import (
    HubertPreTrainedModel,
    HubertModel
)

from data_module import PretrainDataModule
from pretrain_utils import train_step, val_step
from utils import Metric

model_name_or_path = "facebook/hubert-base-ls960"
num_class = 2
epochs = 100
pooling_mode = 'mean'
spilt_rate = [0.6, 0.2, 0.2]
use_amp = True
label_list = ['HC', 'MDD']
batch_size = 16
lr = 1e-3
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
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_class)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
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
        configuration = HubertConfig(
            # hidden_size=512,
            # num_attention_heads=8,
            # num_hidden_layers=6,
        )
        self.hubert = HubertModel(configuration)
        self.classifier = HubertClassificationHead(config)

        self.init_weights()

    def forward(
            self,
            input_values,
    ):
        outputs = self.hubert(
            input_values
        )
        hidden_states = outputs[0]
        hidden_states = merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        return logits


dataset = PretrainDataModule(spilt_rate=spilt_rate)
dataset.setup()
seq_len = dataset.get_seq_len()

train_num = dataset.train_num
val_num = dataset.val_num
test_num = dataset.test_num

train_loader = dataset.train_dataloader(batch_size=batch_size)
val_loader = dataset.val_dataloader(batch_size=batch_size)
# print("load")
model = HubertForSpeechClassification.from_pretrained(
    model_name_or_path,
    config=config,
)
# pretrain_model = torch.load("models/hubert_ser_best.pt")
# model.load_state_dict(pretrain_model.state_dict())
parameter = []
model.gradient_checkpointing_enable()
for name, param in model.named_parameters():  # 仅训练classifier.dense和classifier.out_proj
    if "hubert" in name:
        param.requires_grad = False
        # parameter.append({"params": param, "lr": 0.2 * lr})
    else:
        parameter.append({'params': param, 'lr': lr})

optimizer = torch.optim.AdamW(parameter, lr=lr, betas=(beta1, beta2), weight_decay=0.2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.3)
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
    for step, (bx, by) in enumerate(train_loader):
        correct_num, loss_v = train_step(model, optimizer, bx, by, use_amp, scaler)
        train_correct += correct_num.cpu().numpy()
        train_loss += loss_v
    model.eval()
    with torch.no_grad():
        for step, (vx, vy) in enumerate(val_loader):
            correct_num, loss_v = val_step(model, vx, vy, use_amp)
            val_correct += correct_num.cpu().numpy()
            val_loss += loss_v
    train_acc = train_correct / train_num
    train_loss = train_loss / (math.ceil(train_num / batch_size))
    val_acc = val_correct / val_num
    val_loss = val_loss / (math.ceil(train_num / batch_size))
    metric.train_acc.append(train_acc)
    metric.train_loss.append(train_loss)
    metric.val_acc.append(val_acc)
    metric.val_loss.append(val_loss)
    scheduler.step()
    print("epoch: {}, train_accuracy: {:.3f}\t train_loss: {:.4f}; \t val_accuracy: {:.3f}\t val_loss: {:.4f}".format(
        epoch + 1, train_acc, train_loss, val_acc, val_loss))
    if val_acc > best_val_accuracy:
        torch.save(model, "./models/hubert_ser_best.pt")
        best_val_accuracy = val_acc
        metric.best_val_acc[0] = best_val_accuracy
        metric.best_val_acc[1] = train_acc
        print(f"best val accuracy: {best_val_accuracy}")
np.save("results/hubert.npy", metric.item())
torch.save(model, "./models/hubert_ser_final.pt")

# cm = confusion_matrix(model.test_pred, model.test_true, labels=np.arange(num_class))
# # fig = plot_matrix(cm, labels_name=RAVDESS_LABELS, model_name="test_RAVDESS")
# # logger.log_graph("model", model)
# print(cm)
