import torch.nn as nn
import torch
from einops.layers.torch import Rearrange

from config import Args
from utils import load_loader, Metric, cal_seq_len


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, pool_size):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, padding="same")
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool1d(pool_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_dim, out_dim, scale_factor):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, padding="same")
        self.upsample = nn.Upsample(scale_factor)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.ReLU()


class Encoder(nn.Module):
    def __init__(self, arg: Args):
        super(Encoder, self).__init__()
        self.conv1 = ConvBlock(arg.feature_dim, 64, 2)
        self.conv2 = ConvBlock(64, 64, 2)
        self.conv3 = ConvBlock(64, 128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class Generator(nn.Module):
    def __init__(self, arg: Args):
        super(Generator, self).__init__()
        self.upsample1 = UpsampleBlock(128, 64, 2)
        self.upsample2 = UpsampleBlock(64, 64, 2)
        self.upsample3 = UpsampleBlock(64, arg.feature_dim, 2)

    def forward(self, x):
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        return x


class Classifier(nn.Module):
    def __init__(self, arg: Args):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            Rearrange("N C L -> N (C L)"),
            nn.Linear(cal_seq_len(arg.seq_len, 8) * 128, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(100, 4),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, arg):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=arg.filters, out_channels=arg.filters, kernel_size=3, padding="same"),
            nn.BatchNorm1d(arg.filters),
            # nn.MaxPool1d(2),
            nn.ReLU(),
            nn.Conv1d(in_channels=arg.filters, out_channels=arg.filters, kernel_size=3, padding="same"),
            nn.BatchNorm1d(arg.filters),
            # nn.MaxPool1d(2),
            nn.ReLU(),
            nn.Conv1d(in_channels=arg.filters, out_channels=arg.filters, kernel_size=3, padding="same"),
            nn.BatchNorm1d(arg.filters),
            # nn.MaxPool1d(2),
            nn.ReLU(),
            Rearrange("N C L -> N (C L)"),
            nn.Linear(arg.seq_len * arg.filters, 100),
            nn.Linear(100, 5),  # 5 = 4(情绪类别) + 1
        )

    def forward(self, x):
        x = self.net(x)
        return x


class ADDiTrainer:
    def __init__(self, arg: Args, src_dataset_name, tgt_dataset_name):
        self.src_dt = src_dataset_name
        self.tgt_dt = tgt_dataset_name
        self.src_train, self.src_val = load_loader(self.src_dt, [0.8, 0.2], arg.batch_size, arg.random_seed, "V1")
        self.tgt_train, self.tgt_val = load_loader(self.tgt_dt, [0.8, 0.2], arg.batch_size, arg.random_seed, "V1")
        self.mini_seq_len = min(self.src_train.dataset.dataset.x.shape[-1], self.tgt_train.dataset.dataset.x.shape[-1])
        self.lr = arg.lr
        self.weight_decay = arg.weight_decay
        self.batch_size = arg.batch_size
        self.iteration = 1000
        self.pretrain_epochs = 50
        self.fine_tune_epochs = 20
        self.fine_tune_num = 300
        self.pretrain_best_path = f"models/ADDi/pretrain_best.pt"
        self.pretrain_final_path = f"models/ADDi/pretrain_final.pt"
        self.tgt_best_path = f"models/ADDi/tgt_encoder_best.pt"
        self.tgt_final_path = f"models/ADDi/tgt_encoder_final.pt"
        self.critic_best_path = f"models/ADDi/critic_best.pt"
        self.critic_final_path = f"models/ADDi/critic_final.pt"
        arg.seq_len = self.mini_seq_len
        self.arg = arg
        self.criterion = nn.CrossEntropyLoss()
        self.metric = Metric()

    def pretext(self):
        x_no = torch.randn([])
        pass
    def train(self):
        print()