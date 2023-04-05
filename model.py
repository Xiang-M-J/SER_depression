import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from CNN import CausalConv, WeightLayer, Temporal_Aware_Block
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


class TIM_Attention(nn.Module):
    """
    TIM网络的注意力机制
    """
    def __init__(self, args: Args):
        super(TIM_Attention, self).__init__()
        self.Wq = nn.Sequential(
            nn.Conv2d(in_channels=args.filters, out_channels=args.d_qkv, kernel_size=1, padding="same"),
            Rearrange("N C L H -> N H L C")
            # [batch_size, feature_dim, seq_len, n_head] -> [batch_size, n_head, seq_len, feature_dim ]
        )
        self.Wk = nn.Sequential(
            nn.Conv2d(in_channels=args.filters, out_channels=args.d_qkv, kernel_size=1, padding="same"),
            Rearrange("N C L H -> N H L C")
        )
        self.Wv = nn.Sequential(
            nn.Conv2d(in_channels=args.filters, out_channels=args.d_qkv, kernel_size=1, padding="same"),
            Rearrange("N C L H -> N H L C")
        )
        self.score_flatten = nn.Sequential(
            Rearrange("N H L C -> N C L H"),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=args.d_qkv, out_channels=args.filters, kernel_size=1, padding="same"),
            Rearrange("N H L C -> N L (H C)")
            # nn.BatchNorm2d(args.filters)
        )
        self.dropout = nn.Dropout(0.1)
        self.x_flatten = Rearrange("N C L H -> N L (H C)")
        self.d_model = (args.dilation * args.d_qkv)
        # self.layer_norm = nn.LayerNorm([args.filters, args.seq_len, args.dilation])
        self.norm = nn.LayerNorm(args.dilation * args.filters)
        self.fc = nn.Sequential(
            nn.Linear(in_features=args.filters * args.dilation, out_features=args.d_ff),
            # nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(in_features=args.d_ff, out_features=args.filters * args.dilation),
            nn.ReLU()
        )

    def forward(self, x):
        query = self.Wq(x)
        key = self.Wk(x)
        value = self.Wv(x)
        attn = torch.matmul(query, key.transpose(-1, -2)) / (np.sqrt(self.d_model))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        score = torch.matmul(attn, value)
        score = self.norm(self.score_flatten(score) + self.x_flatten(x))
        score = self.norm(self.fc(score) + score)
        return score  # shape: [N L H*C]


class AT_TIM(nn.Module):
    """
    基于注意力机制的TIM，用注意力机制取代了weight layer，也可以取代Transformer
    """
    def __init__(self, args: Args):
        super(AT_TIM, self).__init__()
        self.name = "AT_TIM"
        self.dilation_layer = nn.ModuleList([])
        self.conv = (nn.Conv1d(in_channels=args.feature_dim, out_channels=args.filters, kernel_size=1, dilation=1,
                               padding=0))
        if args.dilation is None:
            args.dilation = 8
        for i in [2 ** i for i in range(args.dilation)]:
            self.dilation_layer.append(
                Temporal_Aware_Block(feature_dim=args.filters, filters=args.filters, kernel_size=args.kernel_size,
                                     dilation=i, dropout=args.drop_rate)
            )
        self.drop = nn.Dropout(p=args.drop_rate)

        self.attn = TIM_Attention(args)
        # self.weight = WeightLayer(args.dilation)
        self.classifier = nn.Sequential(
            # nn.AdaptiveAvgPool2d((args.seq_len, 1)),
            # Rearrange("N C L H -> N C (L H)"),
            # nn.AdaptiveAvgPool1d(1),
            nn.Linear(in_features=args.filters * args.dilation, out_features=1),
            nn.Dropout(0.1),
            Rearrange('N C L -> N (C L)'),
            nn.Linear(in_features=args.seq_len, out_features=args.num_class)
        )

    def forward(self, x):
        x_f = x
        x_b = torch.flip(x, dims=[-1])
        x_f = self.conv(x_f)
        x_b = self.conv(x_b)
        skip_stack = None
        skip_out_f = x_f
        skip_out_b = x_b
        for layer in self.dilation_layer:
            skip_out_f = layer(skip_out_f)
            skip_out_b = layer(skip_out_b)
            skip_temp = torch.add(skip_out_f, skip_out_b)
            skip_temp = skip_temp.unsqueeze(-1)
            if skip_stack is None:
                skip_stack = skip_temp
            else:
                skip_stack = torch.cat((skip_stack, skip_temp), dim=-1)
                # skip_stack shape: [batch_size, feature_dim, seq_len, dilation]
        x = self.attn(skip_stack)
        # x = self.weight(x)
        x = self.classifier(x)
        return x


class Prepare(nn.Module):
    """
    mfcc -> Transformer
    """
    def __init__(self, args: Args, hidden_dim=128):
        super(Prepare, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=args.feature_dim, out_channels=hidden_dim, kernel_size=1, padding="same")
        self.bn1 = nn.BatchNorm1d(num_features=hidden_dim)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=args.d_model, kernel_size=1, padding="same")
        self.bn2 = nn.BatchNorm1d(num_features=args.d_model)
        self.act = nn.ReLU()
        self.rearrange = Rearrange("N C L -> N L C")

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return self.rearrange(x)


class Middle(nn.Module):
    """
    Transformer -> TIM
    """
    def __init__(self, args: Args):
        super(Middle, self).__init__()
        self.rearrange = Rearrange("N L C -> N C L")
        self.bn = nn.BatchNorm1d(args.d_model)
        self.conv = nn.Conv1d(in_channels=args.d_model, out_channels=args.feature_dim, kernel_size=1, padding="same")

    def forward(self, x):
        return self.conv(self.bn(self.rearrange(x)))


class Transformer_DeltaTIM(nn.Module):
    """
    Transformer + 差分TIM（用后一级的TAB输出减去前一级的TAB输出，共7个差分再加上最后一级的TAB输出共计8个送入weight layer）
    """
    def __init__(self, args: Args):
        super(Transformer_DeltaTIM, self).__init__()
        self.name = "Transformer_DeltaTIM"
        self.prepare = Prepare(args)

        encoderLayer = TransformerEncoderLayer(args.d_model, args.n_head, args.d_ff, args.drop_rate,
                                               batch_first=True)
        self.generalFeatureExtractor = TransformerEncoder(
            encoderLayer, num_layers=args.n_layer
        )
        self.middle = Middle(args)

        self.dilation_layer = nn.ModuleList([])
        self.conv = (nn.Conv1d(in_channels=args.feature_dim, out_channels=args.filters, kernel_size=1, dilation=1,
                               padding=0))
        if args.dilation is None:
            args.dilation = 8

        for i in [2 ** i for i in range(args.dilation)]:
            self.dilation_layer.append(
                Temporal_Aware_Block(feature_dim=args.filters, filters=args.filters, kernel_size=args.kernel_size,
                                     dilation=i, dropout=args.drop_rate)
            )
        self.drop = nn.Dropout(p=args.drop_rate)
        self.weight = WeightLayer(args.dilation)
        # self.weight = WeightLayer(2 * args.dilation - 1)
        self.Classifier = nn.Sequential(
            Rearrange('N C L D -> N (C L D)'),
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
        x = self.conv(x)
        skip_stack = None
        delta_stack = None
        skip_out = x
        for layer in self.dilation_layer:
            skip_out = layer(skip_out)
            if skip_stack is None:
                skip_stack = skip_out.unsqueeze(0)
            else:
                skip_stack = torch.cat((skip_stack, skip_out.unsqueeze(0)), dim=0)
        for i in range(len(skip_stack) - 1):
            if delta_stack is None:
                skip_temp = skip_stack[i + 1] - skip_stack[i]
                delta_stack = skip_temp.unsqueeze(-1)
            else:
                skip_temp = skip_stack[i + 1] - skip_stack[i]
                delta_stack = torch.cat((delta_stack, skip_temp.unsqueeze(-1)), dim=-1)
        delta_stack = torch.cat([delta_stack, skip_out.unsqueeze(-1)], dim=-1)
        # delta_stack = torch.cat([delta_stack, skip_stack.permute(1, 2, 3, 0)], dim=-1)
        x = self.weight(delta_stack)
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
        if mask is not None:
            x = self.generalFeatureExtractor(x, mask)
        else:
            x = self.generalFeatureExtractor(x)
        x = self.Classifier(x)
        return x


class MLTransformer_TIM(nn.Module):  # 可能由于Transformer的层数不深，导致ML的意义不大
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


class SET(nn.Module):
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


if __name__ == "__main__":
    args = Args()
    # x = torch.rand([16, 39, 313]).cuda()
    # model = CNN_Transformer(args).cuda()
    # input_ = torch.LongTensor([[1, 2, 4, 5, 6, 7]])
    # print(nn.Embedding(10, args.feature_dim)(input_).shape)
    # y, scores = model(x)
    # print(y.shape, scores.shape)
    model = Transformer_DeltaTIM(args).cuda()
    x = torch.rand([4, 39, 313]).cuda()
    y = model(x)
    print(y.shape)
    pass
