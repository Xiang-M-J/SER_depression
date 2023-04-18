import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.utils import weight_norm

from CNN import CausalConv, WeightLayer
from Transformer import PositionEncoding
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


class SET(nn.Module):
    """
    speech emotion Transformer（论文版本）
    """

    def __init__(self, args: Args):
        super(SET, self).__init__()
        self.name = "SET"
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


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, arg: Args):
        super(TemporalConvNet, self).__init__()
        self.name = "TCN"
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=arg.feature_dim, out_channels=arg.filters, kernel_size=1, dilation=1, padding=0),
            nn.BatchNorm1d(arg.filters),
            nn.ReLU()
        )
        layers = []
        for i in range(arg.dilation):
            dilation_size = 2 ** i
            layers += [TemporalBlock(arg.feature_dim, arg.filters, arg.kernel_size, stride=1, dilation=dilation_size,
                                     padding=(arg.kernel_size - 1) * dilation_size, dropout=arg.drop_rate)]

        self.network = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange('N C L -> N (C L)'),
            nn.Linear(in_features=arg.filters, out_features=arg.num_class)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.network(x)
        x = self.classifier(x)
        return x


class LSTM(nn.Module):
    def __init__(self, arg: Args):  # input [batch_size, time_step, feature_dim]
        super(LSTM, self).__init__()
        self.name = "LSTM"
        self.lstm = nn.Sequential(
            Rearrange("N C L -> N L C"),
            nn.LSTM(input_size=arg.feature_dim, hidden_size=256, batch_first=True, bidirectional=False)
        )

        self.classifier = nn.Sequential(
            Rearrange("N L C -> N C L"),
            nn.Linear(arg.seq_len, 1),
            Rearrange("N C L -> N (C L)"),
            nn.Linear(256, arg.num_class)
        )

    def forward(self, x):
        self.lstm[1].flatten_parameters()  # 调用flatten_parameters让parameter的数据存放成连续的块，提高内存的利用率和效率
        RNN_out, (_, _) = self.lstm(x)  # if bidirectional=True，输出将包含序列中每个时间步的正向和反向隐藏状态的串联
        x = self.classifier(RNN_out)
        return x


class MLP(nn.Module):
    """
    测试集准确率： 77.528%
    """

    def __init__(self, args: Args):
        super(MLP, self).__init__()
        self.name = "MLP"
        self.prepare = nn.Sequential(
            nn.Conv1d(in_channels=args.feature_dim, out_channels=args.filters, kernel_size=1, dilation=1, padding=0),
            nn.BatchNorm1d(args.filters),
            nn.ReLU(),
        )
        self.classifier = self.classifier = nn.Sequential(
            # nn.Linear(in_features=args.filters * args.dilation, out_features=1),
            # nn.Dropout(0.3),
            # Rearrange('N C L -> N (C L)'),
            # nn.Linear(in_features=args.seq_len, out_features=args.num_class)
            # Rearrange('N L C -> N C L'),
            nn.AdaptiveAvgPool1d(256),
            nn.Linear(in_features=256, out_features=1),
            nn.Dropout(0.3),
            Rearrange('N C L -> N (C L)'),
            nn.Linear(in_features=args.filters, out_features=args.num_class)
        )

    def forward(self, x, mask=None):
        x = self.prepare(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    arg = Args()
    model = LSTM(arg=arg)
    x = torch.rand([4, 39, 313])
    y = model(x)
    print(y.shape)
