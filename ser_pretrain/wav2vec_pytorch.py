import math

import numpy as np
import pytorch_lightning.callbacks as plc
import torch
import torch.nn as nn
import torchaudio
from sklearn.metrics import confusion_matrix
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from preprocess.process_utils import get_files
from utils import accuracy_cal
from utils import myWavLoader

epochs = 50
lr = 1e-3
beta1 = 0.93
beta2 = 0.98
step_size = 30
gamma = 0.3
# seq_len = 64000
batch_size = 8
num_class = 2
spilt_rate = [0.8, 0.1, 0.1]
use_amp = True


def load_callbacks():
    callbacks = [plc.EarlyStopping(
        monitor='val_acc',
        mode='max',
        patience=20,
        min_delta=0.001
    ), plc.ModelCheckpoint(
        monitor='val_acc',
        filename='best-{epoch:02d}-{val_acc:.3f}',
        save_top_k=1,
        mode='max',
        save_last=True
    ), plc.LearningRateMonitor(
        logging_interval='epoch')]
    return callbacks


def cal_hidden(seq_len):
    h1 = math.floor((seq_len - 10) / 5 + 1)
    h2 = math.floor((h1 - 3) / 2 + 1)
    h3 = math.floor((h2 - 3) / 2 + 1)
    h4 = math.floor((h3 - 3) / 2 + 1)
    h5 = math.floor((h4 - 3) / 2 + 1)
    h6 = math.floor((h5 - 2) / 2 + 1)
    h7 = math.floor((h6 - 2) / 2 + 1)
    return h7


class Wav2vec(nn.Module):
    def __init__(self, seq_len, num_class):
        super(Wav2vec, self).__init__()
        # self.pool = nn.MaxPool1d(2)
        # configuration = Wav2Vec2Config(num_attention_heads=6, num_hidden_layers=6)
        self.encoder = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model()

        # output torch.Size([batch_size, seq_len, feature_dim])  feature_dim=32
        # self.bn = nn.BatchNorm1d(256)  # input [batch_size feature_dim, seq_len]
        hidden_size = cal_hidden(seq_len=seq_len)
        self.fc1 = nn.Linear(hidden_size, 1)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_class)

    def forward(self, x):
        # x = self.pool(x)
        x = self.encoder(x)[0]
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


class MODMADataModule():
    def __init__(self, spilt_rate=None, data_dir: str = "preprocess/MODMA", duration: int = 10):
        super().__init__()
        if spilt_rate is None:
            spilt_rate = [0.8, 0.1, 0.1]
        self.data_dir = data_dir
        self.duration = duration
        self.num_classes = 2
        self.spilt_rate = spilt_rate
        self.files = np.array(get_files(self.data_dir))
        self.train_num = int(len(self.files) * spilt_rate[0])
        self.val_num = int(len(self.files) * spilt_rate[1])
        self.test_num = len(self.files) - self.train_num - self.val_num

    def setup(self, stage=None):
        random_index = np.random.permutation(len(self.files))
        train_wavs = self.files[random_index[:self.train_num]]
        val_wavs = self.files[random_index[self.train_num:self.train_num + self.val_num]]
        test_wavs = self.files[random_index[self.train_num + self.val_num:]]
        self.train_dataset = myWavLoader(train_wavs)
        self.val_dataset = myWavLoader(val_wavs)
        self.test_dataset = myWavLoader(test_wavs)

    def get_seq_len(self):
        return self.train_dataset.get_seq_len()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=batch_size)


loss_fn = torch.nn.CrossEntropyLoss()
dataset = MODMADataModule(spilt_rate=spilt_rate)
dataset.setup()
seq_len = dataset.get_seq_len()

train_num = dataset.train_num
val_num = dataset.val_num
test_num = dataset.test_num

train_loader = dataset.train_dataloader()
val_loader = dataset.val_dataloader()

model = Wav2vec(seq_len=seq_len, num_class=num_class)
for name, param in model.named_parameters():
    if "encoder" in name:
        param.requires_grad = False

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(beta1, beta2))
loss_fn = torch.nn.CrossEntropyLoss()

torch.set_float32_matmul_precision('medium')

if use_amp:
    scaler = GradScaler()

if torch.cuda.is_available():
    model = model.cuda()
    loss_fn = loss_fn.cuda()
for epoch in range(epochs):
    model.train()
    train_correct = 0
    train_loss = 0
    val_correct = 0
    val_loss = 0
    for step, (bx, by) in enumerate(train_loader):
        if torch.cuda.is_available():
            bx = bx.cuda()
            by = by.cuda()
        optimizer.zero_grad()
        if use_amp:
            with autocast():
                output = model(bx)
                loss = loss_fn(output, by)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(bx)
            loss = loss_fn(output, by)
            loss.backward()
            optimizer.step()
        train_correct += accuracy_cal(output, by)
        train_loss += loss.data.item()
    model.eval()
    with torch.no_grad():
        for step, (vx, vy) in enumerate(val_loader):
            if torch.cuda.is_available():
                vx = vx.cuda()
                vy = vy.cuda()
            if use_amp:
                with autocast():
                    output = model(vx)
                    loss = loss_fn(output, vy)
            else:
                output = model(vx)
                loss = loss_fn(output, vy)
            val_correct += accuracy_cal(output, vy)
            val_loss += loss.data.item()
    print("epoch: {}, train_accuracy: {:.3f}\t train_loss: {:.4f}; \t val_accuracy: {:.3f}\t val_loss: {:.4f}".format(
        epoch + 1, train_correct / train_num, train_loss / (math.ceil(train_num / batch_size)),
        val_correct / val_num, train_loss / (math.ceil(val_num / batch_size))))

cm = confusion_matrix(model.test_pred, model.test_true, labels=np.arange(num_class))
# fig = plot_matrix(cm, labels_name=RAVDESS_LABELS, model_name="test_RAVDESS")
# logger.log_graph("model", model)
print(cm)
