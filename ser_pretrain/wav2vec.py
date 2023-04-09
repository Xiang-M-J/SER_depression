import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.cuda.amp import GradScaler
from transformers import Wav2Vec2ForPreTraining

from data_module import MODMADataModule
from pretrain_utils import cal_hidden, train_step, val_step

epochs = 50
lr = 1e-3
beta1 = 0.93
beta2 = 0.98
step_size = 30
gamma = 0.3
batch_size = 8
num_class = 2
spilt_rate = [0.8, 0.1, 0.1]
use_amp = True


class Wav2vec(nn.Module):
    def __init__(self, seq_len, num_class):
        super(Wav2vec, self).__init__()
        # self.pool = nn.MaxPool1d(2)
        # configuration = Wav2Vec2Config(num_attention_heads=6, num_hidden_layers=6)
        self.encoder = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base")
        # output torch.Size([batch_size, seq_len, feature_dim])  feature_dim=32
        # self.bn = nn.BatchNorm1d(256)  # input [batch_size feature_dim, seq_len]
        hidden_size = cal_hidden(seq_len=seq_len)
        self.fc1 = nn.Linear(hidden_size, 1)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_class)

    def forward(self, x):
        # x = self.pool(x)
        x = self.encoder(x)['projected_quantized_states']
        x = self.drop(x)
        x = x.permute(0, 2, 1)
        # x = self.bn(x)
        # x = self.pool(x)
        # x = x.reshape(x.size(0), 4 * 32)
        x = self.fc1(x)
        # x = x.reshape(x.size(0), 32)
        x = x.squeeze(-1)

        x = self.fc2(x)
        # x = self.act(x)
        return x


dataset = MODMADataModule(spilt_rate=spilt_rate)
dataset.setup()
seq_len = dataset.get_seq_len()

train_num = dataset.train_num
val_num = dataset.val_num
test_num = dataset.test_num

train_loader = dataset.train_dataloader(batch_size=batch_size)
val_loader = dataset.val_dataloader(batch_size=batch_size)

model = Wav2vec(seq_len=seq_len, num_class=num_class)
for name, param in model.named_parameters():
    if "encoder" in name:
        param.requires_grad = False

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(beta1, beta2))

torch.set_float32_matmul_precision('medium')

if use_amp:
    scaler = GradScaler()

if torch.cuda.is_available():
    model = model.cuda()
for epoch in range(epochs):
    model.train()
    train_correct = 0
    train_loss = 0
    val_correct = 0
    val_loss = 0
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for step, (bx, by) in enumerate(train_loader):
            correct_num, loss_v = train_step(model, optimizer, bx, by, use_amp, scaler)
            train_correct += correct_num
            train_loss += loss_v
    print(prof)
    model.eval()
    with torch.no_grad():
        for step, (vx, vy) in enumerate(val_loader):
            correct_num, loss_v = val_step(model, vx, vy, use_amp)
            val_correct += correct_num
            val_loss += loss_v
    print("epoch: {}, train_accuracy: {:.3f}\t train_loss: {:.4f}; \t val_accuracy: {:.3f}\t val_loss: {:.4f}".format(
        epoch + 1, train_correct / train_num, train_loss / (math.ceil(train_num / batch_size)),
        val_correct / val_num, train_loss / (math.ceil(val_num / batch_size))))

# cm = confusion_matrix(model.test_pred, model.test_true, labels=np.arange(num_class))
# fig = plot_matrix(cm, labels_name=RAVDESS_LABELS, model_name="test_RAVDESS")
# logger.log_graph("model", model)
# print(cm)
