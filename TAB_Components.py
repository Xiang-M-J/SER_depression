import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from Transformer import feedForward
from attention import MultiHeadAttention, AFTLocalAttention, AFTSimpleAttention
from blocks import Temporal_Aware_Block
from config import Args


class TAB_Attention_SE(nn.Module):
    """
    SE
    """

    def __init__(self, args: Args):
        super(TAB_Attention_SE, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool1d(1)  # N C L -> N C 1
        self.pool2 = nn.AdaptiveMaxPool1d(1)
        self.channel = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=16, kernel_size=1),
            # nn.BatchNorm1d(args.filters),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=128, kernel_size=1),
            # nn.BatchNorm1d(args.filters),
            nn.Sigmoid(),
        )

    def forward(self, x, mask=None):
        attn1 = self.pool1(x)
        attn2 = self.pool2(x)
        attn = self.channel(attn1 + attn2)
        x = x * attn
        # x_compress = torch.cat([torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)], dim=1)
        # attn = self.spatial(x_compress)
        # x = x * attn
        return x


class TAB_Attention_MH(nn.Module):
    """
    多头注意力机制
    """
    def __init__(self, args: Args):
        super(TAB_Attention_MH, self).__init__()
        self.in_proj = nn.Sequential(
            Rearrange("N C L -> N L C")
        )
        # self.attn = TransformerEncoderLayer(128, 4, 1024, 0.1, batch_first=True)
        self.attn = MultiHeadAttention(128, 4, 32, 32, 32)
        # self.attn = MultiHeadAttention(39, 3, 13, 13, 13)
        # self.ff = feedForward(256, 1024)
        self.out_proj = nn.Sequential(
            Rearrange("N L C -> N C L"),
        )

    def forward(self, x, mask=None):
        x = self.in_proj(x)
        if mask is not None:
            x, _ = self.attn(x, x, x, mask)
        else:
            x, _ = self.attn(x, x, x, mask)
        return self.out_proj(x)


class TAB_Attention_AF(nn.Module):
    """
    AFT_Local
    """
    def __init__(self, args: Args):
        super(TAB_Attention_AF, self).__init__()
        self.in_proj = nn.Sequential(
            Rearrange("N C L -> N L C")
        )
        self.attn = AFTLocalAttention(128, args.seq_len, 64)
        # self.ff = feedForward(128, 1024)
        self.out_proj = nn.Sequential(
            Rearrange("N L C -> N C L")
        )

    def forward(self, x, mask=None):
        x = self.in_proj(x)
        x, attn = self.attn(x, x, x, mask)
        # x = self.ff(x)
        return self.out_proj(x)


class TAB_Attention_AFS(nn.Module):
    """
    AFT_Simple
    """
    def __init__(self, args: Args):
        super(TAB_Attention_AFS, self).__init__()
        self.in_proj = nn.Sequential(
            Rearrange("N C L -> N L C")
        )
        self.attn = AFTSimpleAttention(128)  # AFTLocal的效果更好
        # self.ff = feedForward(128, 1024)
        self.out_proj = nn.Sequential(
            Rearrange("N L C -> N C L")
        )

    def forward(self, x, mask=None):
        x = self.in_proj(x)
        x, attn = self.attn(x, x, x, mask)
        # x = self.ff(x)
        return self.out_proj(x)


# 各种TAB组件
class TAB(nn.Module):
    """
    原始TAB模块，单向，每一级TAB的输出作为该尺度的输出
    forward(x_f)
    """
    def __init__(self, args: Args, num_layer, index):
        super(TAB, self).__init__()
        self.net = nn.ModuleList([])
        layer_begin = 0 if index == 0 else num_layer[0]
        layer_end = num_layer[0] if index == 0 else sum(num_layer)
        for i in [2 ** i for i in range(layer_begin, layer_end)]:
            self.net.append(
                Temporal_Aware_Block(feature_dim=args.filters, filters=args.filters, kernel_size=args.kernel_size,
                                     dilation=i, dropout=args.drop_rate)
            )

    def forward(self, x):
        stack_layer = []
        for layer in self.net:
            x = layer(x)
            stack_layer.append(x)
        stack_layer = torch.stack(stack_layer, dim=-1)
        return stack_layer, x


class TAB_ADD(nn.Module):
    """
    TIM-Net中的TAB模块，双向，双向两级TAB的输出相加作为该尺度的输出
    forward(x_f, x_b)
    """

    def __init__(self, args: Args, num_layer, index):
        super(TAB_ADD, self).__init__()
        self.net = nn.ModuleList([])
        layer_begin = 0 if index == 0 else num_layer[0]
        layer_end = num_layer[0] if index == 0 else sum(num_layer)
        for i in [2 ** i for i in range(layer_begin, layer_end)]:
            self.net.append(
                Temporal_Aware_Block(feature_dim=args.filters, filters=args.filters, kernel_size=args.kernel_size,
                                     dilation=i, dropout=args.drop_rate)
            )

    def forward(self, x_f, x_b):
        stack_layer = []
        for layer in self.net:
            x_f = layer(x_f)
            x_b = layer(x_b)
            tmp = torch.add(x_f, x_b)
            stack_layer.append(tmp)
        stack_layer = torch.stack(stack_layer, dim=-1)
        return stack_layer, x_f, x_b


class TAB_AT(nn.Module):
    """
    修改版，双向，对双向相加后的数据进行attn作为该尺度的输出
    """

    def __init__(self, args: Args, num_layer, index):
        super(TAB_AT, self).__init__()
        self.in_proj = nn.Sequential(
            nn.Conv1d(args.filters, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            # nn.ReLU(),
        )
        self.net = nn.ModuleList([])
        self.attn = TAB_Attention_MH(args)
        layer_begin = 0 if index == 0 else num_layer[0]
        layer_end = num_layer[0] if index == 0 else sum(num_layer)
        for i in [2 ** i for i in range(layer_begin, layer_end)]:
            self.net.append(
                Temporal_Aware_Block(feature_dim=128, filters=128, kernel_size=args.kernel_size,
                                     dilation=i, dropout=args.drop_rate)
            )
        self.out_proj1 = nn.Sequential(
            nn.Conv1d(128, args.filters, kernel_size=1),
            nn.BatchNorm1d(args.filters),
            # nn.ReLU()
        )
        self.out_proj2 = nn.Sequential(
            nn.Conv2d(128, args.filters, kernel_size=1),
            nn.BatchNorm2d(args.filters),
            # nn.ReLU()
        )

    def forward(self, x_f, x_b, mask=None):
        stack_layer = []
        x_f = self.in_proj(x_f)
        x_b = self.in_proj(x_b)
        for layer in self.net:
            x_f = layer(x_f)
            x_b = layer(x_b)
            tmp = x_f + x_b
            tmp = self.attn(tmp, mask)
            stack_layer.append(tmp)
        stack_layer = torch.stack(stack_layer, dim=-1)

        return self.out_proj2(stack_layer), self.out_proj1(x_f), x_b


class TAB_Conv(nn.Module):
    """
    修改版，双向，双向两级TAB输出级联后进行卷积作为该级的输出
    """

    def __init__(self, args: Args, num_layer, index):
        super(TAB_Conv, self).__init__()
        self.net = nn.ModuleList([])
        self.fusion = nn.ModuleList([])
        layer_begin = 0 if index == 0 else num_layer[0]
        layer_end = num_layer[0] if index == 0 else sum(num_layer)
        for i in [2 ** i for i in range(layer_begin, layer_end)]:
            self.fusion.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=2 * args.filters, out_channels=args.filters, kernel_size=1, padding="same"),
                    nn.BatchNorm1d(args.filters),
                    nn.ReLU(),
                )
            )
            self.net.append(
                Temporal_Aware_Block(feature_dim=args.filters, filters=args.filters, kernel_size=args.kernel_size,
                                     dilation=i, dropout=args.drop_rate)
            )

    def forward(self, x_f, x_b):
        stack_layer = []
        for i, layer in enumerate(self.net):
            x_f = layer(x_f)
            x_b = layer(x_b)
            tmp = self.fusion[i](torch.cat([x_f, x_b], dim=1))
            stack_layer.append(tmp)
        stack_layer = torch.stack(stack_layer, dim=-1)
        return stack_layer, x_f, x_b


class TAB_BiDIFF(nn.Module):
    """
    修改版，双向，双向两级TAB输出均进行差分后相加作为该级的输出
    forward(x_f, x_b)
    """
    def __init__(self, args: Args, num_layer, index):
        super(TAB_BiDIFF, self).__init__()
        self.net = nn.ModuleList([])
        layer_begin = 0 if index == 0 else num_layer[0]
        layer_end = num_layer[0] if index == 0 else sum(num_layer)
        for i in [2 ** i for i in range(layer_begin, layer_end)]:
            self.net.append(
                Temporal_Aware_Block(feature_dim=args.filters, filters=args.filters, kernel_size=args.kernel_size,
                                     dilation=i, dropout=args.drop_rate)
            )

    def forward(self, x_f, x_b):
        stack_layer = []
        last_x_f = None
        last_x_b = None
        for layer in self.net:
            x_f = layer(x_f)
            x_b = layer(x_b)
            if last_x_f is not None:
                stack_layer.append(torch.add(x_f - last_x_f, x_b - last_x_b))
            last_x_f = x_f
            last_x_b = x_b

        stack_layer.append(x_f + x_b)
        stack_layer = torch.stack(stack_layer, dim=-1)

        return stack_layer, x_f, x_b


class AT_TAB(nn.Module):
    """
    修改版，单向，对该级的TAB输出进行attn后作为该级的输出
    forward(self, x_f, x_b, mask=None)
    """
    def __init__(self, args: Args, num_layer, index):
        super(AT_TAB, self).__init__()
        self.in_proj = nn.Sequential(
            nn.Conv1d(args.filters, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            # nn.ReLU(),
        )

        self.net = nn.ModuleList([])
        self.attn = TAB_Attention_MH(args)
        layer_begin = 0 if index == 0 else num_layer[0]
        layer_end = num_layer[0] if index == 0 else sum(num_layer)
        for i in [2 ** i for i in range(layer_begin, layer_end)]:
            self.net.append(
                Temporal_Aware_Block(feature_dim=128, filters=128, kernel_size=args.kernel_size,
                                     dilation=i, dropout=args.drop_rate)
            )
        self.out_proj1 = nn.Sequential(
            nn.Conv1d(128, args.filters, kernel_size=1),
            nn.BatchNorm1d(args.filters),
            # nn.ReLU(),
        )
        self.out_proj2 = nn.Sequential(
            nn.Conv2d(128, args.filters, kernel_size=1),
            nn.BatchNorm2d(args.filters),
            # nn.ReLU()
        )

    def forward(self, x_f, x_b, mask=None):
        stack_layer = []
        x_f = self.in_proj(x_f)
        for layer in self.net:
            x_f = layer(x_f)
            x_f = self.attn(x_f, mask)
            stack_layer.append(x_f)
        stack_layer = torch.stack(stack_layer, dim=-1)
        return self.out_proj2(stack_layer), self.out_proj1(x_f), x_b


class TAB_DIFF(nn.Module):
    """
    修改版，单向，对该级TAB的输出进行差分作为该级的输出
    forward(self, x)
    """
    def __init__(self, args: Args, num_layer, index):
        super(TAB_DIFF, self).__init__()
        self.net = nn.ModuleList([])
        layer_begin = 0 if index == 0 else num_layer[0]
        layer_end = num_layer[0] if index == 0 else sum(num_layer)
        for i in [2 ** i for i in range(layer_begin, layer_end)]:
            self.net.append(
                Temporal_Aware_Block(feature_dim=args.filters, filters=args.filters,
                                     kernel_size=args.kernel_size,
                                     dilation=i, dropout=args.drop_rate)
            )

    def forward(self, x):
        stack_layer = []
        last_x = None
        for layer in self.net:
            x = layer(x)
            if last_x is not None:
                stack_layer.append(x - last_x)
            last_x = x
        stack_layer.append(x)
        stack_layer = torch.stack(stack_layer, dim=-1)
        return stack_layer, x
