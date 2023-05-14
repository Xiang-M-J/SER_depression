import time

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from CNN import CausalConv, WeightLayer
from Transformer import PositionEncoding, Encoder
from TransformerEncoder import TransformerEncoder, TransformerEncoderLayer
from config import Args
from utils import cal_seq_len


class TIM(nn.Module):
    """
    基准网络
    """

    def __init__(self, args: Args):
        super(TIM, self).__init__()
        self.name = "TIM"
        self.extractor = CausalConv(args)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange('N C L -> N (C L)'),
            nn.Linear(in_features=args.filters, out_features=args.num_class)
        )

    def forward(self, x):
        x = self.extractor(x)
        x = self.classifier(x)
        return x


class CNN_Transformer(nn.Module):  # input shape: [N, C, L]
    """落后版本的模型，将自己写的Transformer换成官方的Transformer后也能在迭代一段时候后稳定大概91%"""

    def __init__(self, args: Args):
        super(CNN_Transformer, self).__init__()
        self.name = "CNN_Transformer"
        self.generalFeatureExtractor = CausalConv(args)
        self.middle = nn.Sequential(
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=args.filters, out_channels=args.d_model, kernel_size=1, padding="same"),
            Rearrange("N C L -> N L C")
        )
        encoderLayer = TransformerEncoderLayer(args.d_model, args.n_head, args.d_ff, args.drop_rate,
                                               batch_first=True)
        self.specificFeatureExtractor = TransformerEncoder(
            encoderLayer, num_layers=args.n_layer
        )
        self.Classifier = nn.Sequential(
            Rearrange('N L C -> N C L'),
            nn.Linear(in_features=cal_seq_len(args.seq_len, 2), out_features=1),
            Rearrange('N C L -> N (C L)'),
            nn.Linear(in_features=args.d_model, out_features=args.num_class),
        )

    def forward(self, x, mask=None):
        x = self.generalFeatureExtractor(x)
        x = self.middle(x)
        if mask is not None:
            x = self.specificFeatureExtractor(x, mask)
        else:
            x = self.specificFeatureExtractor(x)
        x = self.Classifier(x)
        return x


class CNN_Transformer_error(nn.Module):  # input shape: [N, C, L]
    """
    自己写的TransformerEncoder不太好使
    """

    def __init__(self, args: Args):
        super(CNN_Transformer_error, self).__init__()
        # self.plus_scores = args.plus_scores
        self.generalFeatureExtractor = CausalConv(args)
        self.middle = nn.Sequential(
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=args.filters, out_channels=args.d_model, kernel_size=1, padding="same")
        )

        self.specificFeatureExtractor = Encoder(args.d_model, args.n_head, args.d_qkv, args.d_qkv, args.d_qkv,
                                                args.d_ff, args.n_layer)
        self.Classifier = nn.Sequential(
            Rearrange('N L C -> N C L'),
            nn.Linear(in_features=cal_seq_len(args.seq_len, 2), out_features=1),
            nn.Dropout(0.2),
            Rearrange('N C L -> N (C L)'),
            nn.Linear(in_features=args.d_model, out_features=args.num_class),
        )

    def forward(self, x):
        x = self.generalFeatureExtractor(x)
        x = self.middle(x)
        x, _ = self.specificFeatureExtractor(x)
        x = self.Classifier(x)
        return x


class CNN_ML_Transformer(nn.Module):
    """
    换了官方的Transformer后，模型性能一个字拉，要靠warmup才能勉强跑出比较好看的成绩，也不容易收敛(100次还是波动幅度大)
    """

    def __init__(self, args: Args):
        super(CNN_ML_Transformer, self).__init__()
        self.name = "CNN_ML_Transformer"
        self.generalFeatureExtractor = CausalConv(args)
        self.middle = nn.Sequential(
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=args.filters, out_channels=args.d_model, kernel_size=1, padding="same")
        )
        self.position_encoding = nn.Sequential(
            Rearrange("N C L -> N L C"),
            PositionEncoding(args.d_model)
        )
        self.specificFeatureExtractor = nn.ModuleList([])

        for i in range(args.n_layer):
            self.specificFeatureExtractor.append(
                TransformerEncoderLayer(args.d_model, args.n_head, args.d_ff, args.drop_rate,
                                        batch_first=True)
            )
        self.weight = WeightLayer(args.n_layer)
        self.Classifier = nn.Sequential(
            Rearrange('N L C W -> N C (L W)'),
            nn.Linear(in_features=cal_seq_len(args.seq_len, 2), out_features=1),
            Rearrange('N C L -> N (C L)'),
            nn.Linear(in_features=args.d_model, out_features=args.num_class),
        )

    def forward(self, x, mask=None):
        x = self.generalFeatureExtractor(x)
        x = self.middle(x)
        x = self.position_encoding(x)
        multi_layer = None
        for layer in self.specificFeatureExtractor:
            if mask is not None:
                x = layer(x, mask)
            else:
                x = layer(x)
            if multi_layer is None:
                multi_layer = x.unsqueeze(-1)
            else:
                multi_layer = torch.cat([multi_layer, x.unsqueeze(-1)], dim=-1)
        x = self.weight(multi_layer)
        x = self.Classifier(x)
        return x


class Transformer_TIM(nn.Module):
    """
    两种改进型（AT_TIM, Transformer_DeltaTIM）的原始模型
    """

    def __init__(self, args: Args):
        super(Transformer_TIM, self).__init__()
        self.name = "Transformer_TIM"
        self.prepare = nn.Sequential(
            nn.Conv1d(in_channels=args.feature_dim, out_channels=128, kernel_size=1, padding="same"),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=args.d_model, kernel_size=1, padding="same"),
            nn.BatchNorm1d(num_features=args.d_model),
            nn.ReLU(),
            Rearrange("N C L -> N L C"),
        )
        encoderLayer = TransformerEncoderLayer(args.d_model, args.n_head, args.d_ff, args.drop_rate,
                                               batch_first=True)
        self.generalFeatureExtractor = TransformerEncoder(
            encoderLayer, num_layers=args.n_layer
        )
        self.middle = nn.Sequential(
            Rearrange("N L C -> N C L"),
            nn.BatchNorm1d(args.d_model),
            nn.Conv1d(in_channels=args.d_model, out_channels=args.feature_dim, kernel_size=1, padding="same"),
        )
        self.specialFeatureExtractor = CausalConv(args)

        self.Classifier = nn.Sequential(
            Rearrange('N C L -> N (C L)'),
            nn.Linear(in_features=args.seq_len * args.filters, out_features=100),
            nn.Dropout(0.2),
            nn.Linear(in_features=100, out_features=args.num_class),
        )

    def forward(self, x, mask=None):
        x = self.prepare(x)
        if mask is not None:
            x = self.generalFeatureExtractor(x, mask)
        else:
            x = self.generalFeatureExtractor(x)
        x = self.middle(x)
        x = self.specialFeatureExtractor(x)
        x = self.Classifier(x)
        return x


class MLTransformer_TIM(nn.Module):
    """
    可能由于Transformer的层数不深，导致ML(multi level)的意义不大
    """

    def __init__(self, args: Args):
        super(MLTransformer_TIM, self).__init__()
        self.name = "MLTransformer_TIM"
        self.prepare = nn.Sequential(
            nn.Conv1d(in_channels=args.feature_dim, out_channels=128, kernel_size=1, padding="same"),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=args.d_model, kernel_size=1, padding="same"),
            nn.BatchNorm1d(num_features=args.d_model),
            nn.ReLU(),
            Rearrange("N C L -> N L C"),
        )
        encoderLayer = TransformerEncoderLayer(args.d_model, args.n_head, args.d_ff, args.drop_rate,
                                               batch_first=True)
        self.generalFeatureExtractor = nn.ModuleList([])
        for _ in range(args.n_layer):
            self.generalFeatureExtractor.append(encoderLayer)
        self.middle = nn.Sequential(
            nn.Linear(in_features=args.n_layer, out_features=1),
            Rearrange("N L C W-> N C (L W)"),
            nn.BatchNorm1d(args.d_model),
            nn.Conv1d(in_channels=args.d_model, out_channels=args.feature_dim, kernel_size=1, padding="same"),
        )
        self.specialFeatureExtractor = CausalConv(args)

        self.Classifier = nn.Sequential(
            Rearrange('N C L -> N (C L)'),
            nn.Linear(in_features=args.seq_len * args.filters, out_features=1000),
            nn.Dropout(0.3),
            nn.Linear(in_features=1000, out_features=args.num_class),
        )

    def forward(self, x, mask=None):
        x = self.prepare(x)
        concat_layer = None
        for layer in self.generalFeatureExtractor:
            if mask is not None:
                x = layer(x, mask)
            else:
                x = layer(x)
            if concat_layer is None:
                concat_layer = x.unsqueeze(-1)
            else:
                concat_layer = torch.cat([concat_layer, x.unsqueeze(-1)], dim=-1)
        x = self.middle(concat_layer)
        x = self.specialFeatureExtractor(x)
        x = self.Classifier(x)
        return x


class Transformer(nn.Module):
    """
    基准网络测试
    """

    def __init__(self, args: Args):
        super(Transformer, self).__init__()
        self.name = "Transformer"
        self.prepare = nn.Sequential(
            nn.Conv1d(in_channels=args.feature_dim, out_channels=128, kernel_size=1, padding="same"),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=args.d_model, kernel_size=1, padding="same"),
            nn.BatchNorm1d(num_features=args.d_model),
            nn.ReLU(),
            Rearrange("N C L -> N L C"),
        )
        self.pe = PositionEncoding(args.d_model, max_len=5000)
        encoderLayer = TransformerEncoderLayer(args.d_model, args.n_head, args.d_ff, args.drop_rate,
                                               batch_first=True)
        self.generalFeatureExtractor = TransformerEncoder(
            encoderLayer, num_layers=args.n_layer
        )
        self.Classifier = nn.Sequential(
            Rearrange("N L C -> N C L"),
            nn.Conv1d(in_channels=args.d_model, out_channels=64, kernel_size=1, padding="same"),
            nn.BatchNorm1d(64),
            Rearrange('N C L -> N (C L)'),
            nn.Linear(in_features=args.seq_len * 64, out_features=1000),
            nn.Dropout(0.3),
            nn.Linear(in_features=1000, out_features=args.num_class),
        )

    def forward(self, x, mask=None):
        x = self.prepare(x)
        # x = self.pe(x)
        if mask is not None:
            x = self.generalFeatureExtractor(x, mask)
        else:
            x = self.generalFeatureExtractor(x)
        x = self.Classifier(x)
        return x


class SET(nn.Module):
    """
    SET (我的版本 简化版本)
    """

    def __init__(self, args: Args):
        super(SET, self).__init__()
        self.name = "SET"
        self.scale_depth = nn.Conv1d(in_channels=args.feature_dim, out_channels=args.d_model, kernel_size=1)
        self.bn = nn.BatchNorm1d(args.d_model)  # bn 可以有效加快收敛速度
        self.position_encoding = nn.Sequential(
            Rearrange("N C L -> N L C"),
            PositionEncoding(args.d_model)
        )
        self.encoder = nn.ModuleList([])

        for i in range(args.n_layer):
            self.encoder.append(
                TransformerEncoderLayer(args.d_model, args.n_head, args.d_ff, args.drop_rate,
                                        batch_first=True)
            )
        # self.bn = nn.BatchNorm2d(args.n_layer)
        self.weight = WeightLayer(args.n_layer)
        self.classifier = nn.Sequential(
            Rearrange("N L C W -> N C (L W)"),
            nn.AdaptiveAvgPool1d(1),
            Rearrange("N C L -> N (C L)"),
            nn.Linear(args.d_model, args.num_class)
        )

    def forward(self, x, mask=None):
        x = self.scale_depth(x)
        x = self.bn(x)
        x = self.position_encoding(x)
        concat_layer = None
        for layer in self.encoder:
            if mask is not None:
                x = layer(x, mask)
            else:
                x = layer(x)
            if concat_layer is None:
                concat_layer = x.unsqueeze(-1)
            else:
                concat_layer = torch.cat([concat_layer, x.unsqueeze(-1)], dim=-1)
        x = self.weight(concat_layer)
        x = self.classifier(x)
        return x


class convBlock(nn.Module):
    """
    speech emotion Transformer（论文版本）中的卷积块
    """

    def __init__(self, in_dim, out_dim, pool_size, kernel_size=3):
        super(convBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm2d(out_dim),
            nn.MaxPool2d(pool_size)
        )

    def forward(self, x):
        return self.block(x)


class SET_official(nn.Module):
    """
    speech emotion Transformer（论文版本）
    """

    def __init__(self, args: Args):
        super(SET_official, self).__init__()
        self.name = "SET_official"
        self.scale_depth = nn.Conv1d(in_channels=args.feature_dim, out_channels=args.d_model, kernel_size=1)
        self.bn = nn.BatchNorm1d(args.d_model)  # bn 可以有效加快收敛速度
        self.position_encoding = nn.Sequential(
            Rearrange("N C L -> N L C"),
            # PositionEncoding(args.d_model)
        )
        self.encoder = nn.ModuleList([])
        for i in range(args.n_layer):
            self.encoder.append(
                TransformerEncoderLayer(args.d_model, args.n_head, args.d_ff, args.drop_rate,
                                        batch_first=True)
            )
        self.bn2 = nn.BatchNorm2d(args.n_layer)
        self.conv = nn.Sequential(
            convBlock(args.n_layer, 16, pool_size=2, kernel_size=3),
            convBlock(16, 32, pool_size=2, kernel_size=3),
            convBlock(32, 64, pool_size=3, kernel_size=3),
            convBlock(64, 128, pool_size=3, kernel_size=3)
        )
        self.classifier = nn.Sequential(
            Rearrange("N C H W -> N (C H W)"),
            nn.Linear(128 * cal_seq_len(args.d_model, 36) * cal_seq_len(args.seq_len, 36), 100),
            nn.Dropout(0.3),
            nn.Linear(100, args.num_class)
        )

    def forward(self, x, mask=None):
        x = self.scale_depth(x)
        x = self.bn(x)
        x = self.position_encoding(x)
        concat_layer = None
        for layer in self.encoder:
            if mask is not None:
                x = layer(x, mask)
            else:
                x = layer(x)
            if concat_layer is None:
                concat_layer = x.unsqueeze(1)
            else:
                concat_layer = torch.cat([concat_layer, x.unsqueeze(1)], dim=1)
        x = self.bn2(concat_layer)
        x = self.conv(x)
        x = self.classifier(x)
        return x
