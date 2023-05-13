import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from blocks import WeightLayer, Temporal_Aware_Block
from TransformerEncoder import TransformerEncoder, TransformerEncoderLayer
from attention import MultiHeadAttention, AFTLocalAttention, get_attention, AFTSimpleAttention
from Transformer import feedForward

from config import Args
from TAB_Components import TAB, TAB_DIFF, TAB_ADD, TAB_Conv, TAB_AT, TAB_BiDIFF, AT_TAB
from utils import seed_everything, mask_input


class AT_TIM(nn.Module):
    """
    基于注意力机制的TIM，用注意力机制取代了weight layer，也可以取代Transformer
    """

    def __init__(self, args: Args):
        super(AT_TIM, self).__init__()
        self.name = "AT_TIM"
        self.dilation_layer = nn.ModuleList([])
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=args.feature_dim, out_channels=args.filters, kernel_size=1, dilation=1, padding=0),
            nn.BatchNorm1d(args.filters),
            nn.ReLU()
        )
        if args.dilation is None:
            args.dilation = 8
        for i in [2 ** i for i in range(args.dilation)]:
            self.dilation_layer.append(
                Temporal_Aware_Block(feature_dim=args.filters, filters=args.filters,
                                     kernel_size=args.kernel_size, dilation=i, dropout=args.drop_rate)
            )
        self.drop = nn.Dropout(p=args.drop_rate)

        self.attn = get_attention(args)
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

    def forward(self, x, mask=None):
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
        if mask is not None:
            x = self.attn(skip_stack, mask)
        else:
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
        delta_stack = []
        skip_out = None
        now_skip_out = x
        for layer in self.dilation_layer:
            now_skip_out = layer(now_skip_out)
            if skip_out is not None:
                delta_stack.append(now_skip_out - skip_out)
            skip_out = now_skip_out
        delta_stack.append(now_skip_out)
        delta_stack = torch.stack(delta_stack, dim=-1)
        x = self.weight(delta_stack)
        x = self.Classifier(x)
        return x


class AT_DeltaTIM(nn.Module):
    """
    基于注意力的差分TIM
    """

    def __init__(self, args: Args):
        super(AT_DeltaTIM, self).__init__()
        self.name = "AT_DeltaTIM"
        self.dilation_layer = nn.ModuleList([])
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=args.feature_dim, out_channels=args.filters, kernel_size=1, dilation=1, padding=0),
            nn.BatchNorm1d(args.filters),
            nn.ReLU()
        )
        if args.dilation is None:
            args.dilation = 8
        for i in [2 ** i for i in range(args.dilation)]:
            self.dilation_layer.append(
                Temporal_Aware_Block(feature_dim=args.filters, filters=args.filters, kernel_size=args.kernel_size,
                                     dilation=i, dropout=args.drop_rate)
            )
        self.drop = nn.Dropout(p=args.drop_rate)

        self.attn = get_attention(args)

        self.classifier = nn.Sequential(
            # nn.AdaptiveAvgPool2d((args.seq_len, 1)),
            # Rearrange("N C L H -> N C (L H)"),
            # nn.AdaptiveAvgPool1d(1),
            nn.Linear(in_features=args.filters * args.dilation, out_features=1),
            nn.Dropout(0.1),
            Rearrange('N C L -> N (C L)'),
            nn.Linear(in_features=args.seq_len, out_features=args.num_class)
        )

    def forward(self, x, mask=None):
        x = self.conv(x)
        delta_stack = []
        skip_out = None
        now_skip_out = x
        for layer in self.dilation_layer:
            now_skip_out = layer(now_skip_out)
            if skip_out is not None:
                delta_stack.append(now_skip_out - skip_out)
            skip_out = now_skip_out
        delta_stack.append(now_skip_out)
        delta_stack = torch.stack(delta_stack, dim=-1)
        if mask is not None:
            x = self.attn(delta_stack, mask)
        else:
            x = self.attn(delta_stack)
        x = self.classifier(x)
        return x


class AT_DIFF_Block(nn.Module):
    """
    AT + DIFF
    """

    def __init__(self, arg: Args, num_layers=None, is_prepare=False):
        super(AT_DIFF_Block, self).__init__()
        if num_layers is None:
            num_layers = [3, 5]
        self.is_prepare = is_prepare
        if is_prepare:
            self.prepare = nn.Sequential(
                nn.Conv1d(in_channels=arg.feature_dim, out_channels=arg.filters, kernel_size=1, padding=0),
                nn.BatchNorm1d(arg.filters),
                nn.ReLU(),
            )
        self.general = AT_TAB(arg, num_layers, 0)
        self.special = TAB_DIFF(arg, num_layers, 1)
        arg.dilation = num_layers[0] + num_layers[1]
        self.merge = nn.Sequential(
            nn.Linear(arg.dilation, 1),
            Rearrange("N C L H -> N C (L H)"),
            nn.BatchNorm1d(arg.filters),
        )

    def forward(self, x, mask=None):
        if self.is_prepare:
            x_f = self.prepare(x)
            x_b = self.prepare(torch.flip(x, dims=[-1]))
        else:
            x_f = x
            x_b = torch.flip(x, dims=[-1])
        x_1, x_f, _ = self.general(x_f, x_b, mask)
        x_2, x_f = self.special(x_f)
        x = self.merge(torch.cat([x_1, x_2], dim=-1))
        return x


class DIFF_AT_Block(nn.Module):
    """
    DIFF + AT
    """

    def __init__(self, arg: Args, num_layers=None, is_prepare=False):
        super(DIFF_AT_Block, self).__init__()
        if num_layers is None:
            num_layers = [3, 5]
        self.is_prepare = is_prepare
        if is_prepare:
            self.prepare = nn.Sequential(
                nn.Conv1d(in_channels=arg.feature_dim, out_channels=arg.filters, kernel_size=1, padding=0),
                nn.BatchNorm1d(arg.filters),
                nn.ReLU(),
            )
        self.general = TAB_DIFF(arg, num_layers, 0)
        self.special = TAB_AT(arg, num_layers, 1)
        arg.dilation = num_layers[0] + num_layers[1]
        self.merge = nn.Sequential(
            nn.Linear(arg.dilation, 1),
            Rearrange("N C L H -> N C (L H)"),
            nn.BatchNorm1d(arg.filters),
        )

    def forward(self, x, mask=None):
        if self.is_prepare:
            x_f = self.prepare(x)
            x_b = self.prepare(torch.flip(x, dims=[-1]))
        else:
            x_f = x
            x_b = torch.flip(x, dims=[-1])
        x_1, x_f = self.general(x_f)
        x_2, x_f, _ = self.special(x_f, x_b, mask)
        x = self.merge(torch.cat([x_1, x_2], dim=-1))
        return x


class vanilla_Block(nn.Module):
    def __init__(self, arg: Args, num_layers=None, is_prepare=False):
        super(vanilla_Block, self).__init__()
        if num_layers is None:
            num_layers = [3, 5]
        self.is_prepare = is_prepare
        if is_prepare:
            self.prepare = nn.Sequential(
                nn.Conv1d(in_channels=arg.feature_dim, out_channels=arg.filters, kernel_size=1, padding=0),
                nn.BatchNorm1d(arg.filters),
                nn.ReLU(),
            )
        self.general = TAB(arg, num_layers, 0)
        self.special = TAB(arg, num_layers, 1)
        arg.dilation = num_layers[0] + num_layers[1]
        self.merge = nn.Sequential(
            nn.Linear(arg.dilation, 1),
            Rearrange("N C L H -> N C (L H)"),
            nn.BatchNorm1d(arg.filters),
        )

    def forward(self, x, mask=None):
        if self.is_prepare:
            x_f = self.prepare(x)
        else:
            x_f = x
        x_1, x_f = self.block1(x_f)
        x_2, x_f = self.block2(x_f)
        x = self.merge(torch.cat([x_1, x_2], dim=-1))
        return x


class vanilla_DIFF_Block(nn.Module):
    """
    Vanilla + DIFF
    """

    def __init__(self, arg: Args, num_layers=None, is_prepare=False):
        super(vanilla_DIFF_Block, self).__init__()
        if num_layers is None:
            num_layers = [3, 5]
        self.is_prepare = is_prepare
        if is_prepare:
            self.prepare = nn.Sequential(
                nn.Conv1d(in_channels=arg.feature_dim, out_channels=arg.filters, kernel_size=1, padding=0),
                nn.BatchNorm1d(arg.filters),
                nn.ReLU(),
            )
        self.general = TAB(arg, num_layers, 0)
        self.special = TAB_DIFF(arg, num_layers, 1)
        arg.dilation = num_layers[0] + num_layers[1]
        self.merge = nn.Sequential(
            nn.Linear(arg.dilation, 1),
            Rearrange("N C L H -> N C (L H)"),
            nn.BatchNorm1d(arg.filters),
        )

    def forward(self, x, mask=None):
        if self.is_prepare:
            x_f = self.prepare(x)
        else:
            x_f = x
        x_1, x_f = self.block1(x_f)
        x_2, x_f = self.block2(x_f)
        x = self.merge(torch.cat([x_1, x_2], dim=-1))
        return x


class AT_vanilla_Block(nn.Module):
    """
    AT + Vanilla
    """

    def __init__(self, arg: Args, num_layers=None, is_prepare=False):
        super(AT_vanilla_Block, self).__init__()
        if num_layers is None:
            num_layers = [3, 5]
        self.is_prepare = is_prepare
        if is_prepare:
            self.prepare = nn.Sequential(
                nn.Conv1d(in_channels=arg.feature_dim, out_channels=arg.filters, kernel_size=1, padding=0),
                nn.BatchNorm1d(arg.filters),
                nn.ReLU(),
            )
        self.general = AT_TAB(arg, num_layers, 0)
        self.special = TAB(arg, num_layers, 1)
        arg.dilation = num_layers[0] + num_layers[1]
        self.merge = nn.Sequential(
            nn.Linear(arg.dilation, 1),
            Rearrange("N C L H -> N C (L H)"),
            nn.BatchNorm1d(arg.filters),
        )

    def forward(self, x, mask=None):
        if self.is_prepare:
            x_f = self.prepare(x)
            x_b = self.prepare(torch.flip(x, dims=[-1]))
        else:
            x_f = x
            x_b = torch.flip(x, dims=[-1])
        x_1, x_f, _ = self.block1(x_f, x_b, mask)
        x_2, x_f = self.block2(x_f)
        x = self.merge(torch.cat([x_1, x_2], dim=-1))
        return x


class MTCN(nn.Module):
    """
    old result:
    MTCN: 当generalExtractor和specialExtractor使用初始化参数，只训练prepare attn classifier 测试集准确率 89.326%
    MTCN: 当generalExtractor和specialExtractor使用IEMOCAP_V1预训练模型参数，只训练prepare attn classifier 测试集准确率 95.225%
    MTCN: 当generalExtractor和specialExtractor使用IEMOCAP_V2预训练模型参数，只训练prepare attn classifier 测试集准确率 97.753%
    MTCN: 当generalExtractor和specialExtractor使用DAIC_V1预训练模型参数，只训练prepare attn classifier 测试集准确率 94.382%
    """

    def __init__(self, args: Args, num_layers=None):
        super(MTCN, self).__init__()
        self.name = "MTCN"
        if num_layers is None:
            num_layers = [3, 5]
        # 多层特征融合
        # nn.Conv2d
        # self.merge = nn.Sequential(
        #     Rearrange("N C L H -> N H C L"),
        #     nn.Conv2d(in_channels=args.dilation, out_channels=1, kernel_size=1),
        #     nn.BatchNorm2d(1),
        #     nn.Dropout(0.3),
        #     Rearrange("N H C L -> N C L H")
        # )
        # 全局平均池化，可以直接torch.mean(torch.cat([x_1, x_2], dim=-1), dim=-1, keepdim=True)
        # self.merge = nn.Sequential(
        #     Rearrange("N C L H -> N (C L) H"),
        #     nn.AdaptiveAvgPool1d(1),
        #     Rearrange("N (C L) H -> N C L H", C=args.filters)
        # )
        # Linear
        # self.merge = nn.Sequential(
        #     # 加上BatchNorm和Dropout后容易验证集虚高
        #     nn.Linear(args.dilation, 1),
        #     Rearrange("N C L H -> N C (L H)"),
        #     nn.BatchNorm1d(args.filters),
        #     # nn.ReLU(),
        #     # nn.Dropout(0.3)
        # )

        self.extractor = AT_DIFF_Block(args, num_layers, True)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            # nn.Linear(args.seq_len, 1),
            nn.Dropout(0.1),
            Rearrange('N C L -> N (C L)'),
            nn.Linear(in_features=args.filters, out_features=args.num_class)
        )

    def forward(self, x, mask=None):
        # x_f = self.prepare(x)
        # x_b = self.prepare(torch.flip(x, dims=[-1]))
        # x_1, x_f, x_b = self.generalExtractor(x_f, x_b)
        # x_2, _, _ = self.specialExtractor(x_f, x_b)

        # for ADD_DIFF
        # x_f = self.prepare(x)
        # x_b = self.prepare(torch.flip(x, dims=[-1]))
        # x_1, x, _ = self.generalExtractor(x_f, x_b)
        # x_2, x = self.specialExtractor(x)

        # for AT_DIFF
        # x_f = self.prepare(x)
        # x_b = self.prepare(torch.flip(x, dims=[-1]))
        # # x_f = x
        # # x_b = torch.flip(x, dims=[-1])
        # x_1, x, _ = self.generalExtractor(x_f, x_b, mask)
        # x_2, x = self.specialExtractor(x)

        # for ADD_BiDIFF ADD_ADD, ADD_Conv, ADD_AT
        # x_f = self.prepare(x)
        # x_b = self.prepare(torch.flip(x, dims=[-1]))
        # x_1, x_f, x_b = self.generalExtractor(x_f, x_b)
        # x_2, _, _ = self.specialExtractor(x_f, x_b)

        # x = self.merge(torch.cat([x_1, x_2], dim=-1))
        # x = torch.mean(torch.cat([x_1, x_2], dim=-1), dim=-1, keepdim=True)   # 另一种平均池化的方法
        x = self.extractor(x, mask)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    pass
