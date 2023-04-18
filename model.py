import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from CNN import WeightLayer, Temporal_Aware_Block, Temporal_Aware_Block_simple
from TransformerEncoder import TransformerEncoder, TransformerEncoderLayer
from attention import MultiHeadAttention, AFTLocalAttention, get_attention, AFTSimpleAttention
from Transformer import feedForward

from config import Args
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
                Temporal_Aware_Block_simple(feature_dim=args.filters, filters=args.filters,
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
        self.spatial = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, mask=None):
        attn1 = self.pool1(x)
        attn2 = self.pool2(x)
        attn = self.channel(attn1 + attn2)
        x = x * attn
        x_compress = torch.cat([torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)], dim=1)
        attn = self.spatial(x_compress)
        x = x * attn
        return x


class TAB_Attention_MH(nn.Module):
    """
    MH
    """

    def __init__(self, args: Args):
        super(TAB_Attention_MH, self).__init__()
        self.in_proj = nn.Sequential(
            Rearrange("N C L -> N L C")
        )
        # self.attn = TransformerEncoderLayer(128, 4, 1024, 0.1, batch_first=True)
        self.attn = MultiHeadAttention(128, 4, 32, 32, 32)
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
    MH
    """

    def __init__(self, args: Args):
        super(TAB_Attention_AF, self).__init__()
        self.in_proj = nn.Sequential(
            Rearrange("N C L -> N L C")
        )
        self.attn = AFTLocalAttention(128, args.seq_len, 64)
        # self.attn = AFTSimpleAttention(128)   # AFTLocal的效果更好
        self.ff = feedForward(128, 1024)
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
        return stack_layer


class TAB_ADD(nn.Module):
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
    对双向相加后的数据进行attn
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


class AT_TAB(nn.Module):
    """
    对单向进行attn
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


class TAB_Conv(nn.Module):
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


class TAB_DIFF(nn.Module):
    def __init__(self, args: Args, num_layer, index):
        super(TAB_DIFF, self).__init__()
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
        last_x = None
        for layer in self.net:
            x = layer(x)
            if last_x is not None:
                stack_layer.append(x - last_x)
            last_x = x
        stack_layer.append(x)
        stack_layer = torch.stack(stack_layer, dim=-1)

        return stack_layer, x


class TAB_BiDIFF(nn.Module):
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


class MultiTIM(nn.Module):
    """
    old result:
    MultiTIM: 当generalExtractor和specialExtractor使用初始化参数，只训练prepare attn classifier 测试集准确率 89.326%
    MultiTIM: 当generalExtractor和specialExtractor使用IEMOCAP_V1预训练模型参数，只训练prepare attn classifier 测试集准确率 95.225%
    MultiTIM: 当generalExtractor和specialExtractor使用IEMOCAP_V2预训练模型参数，只训练prepare attn classifier 测试集准确率 97.753%
    MultiTIM: 当generalExtractor和specialExtractor使用DAIC_V1预训练模型参数，只训练prepare attn classifier 测试集准确率 94.382%
    """

    def __init__(self, args: Args, num_layers=None):
        super(MultiTIM, self).__init__()
        self.name = "MultiTIM"
        if num_layers is None:
            num_layers = [3, 5]
        self.prepare = nn.Sequential(
            # nn.BatchNorm1d(args.feature_dim),
            nn.Conv1d(in_channels=args.feature_dim, out_channels=args.filters, kernel_size=1, dilation=1, padding=0),
            nn.BatchNorm1d(args.filters),
            nn.ReLU(),
        )
        self.generalExtractor = AT_TAB(args, num_layer=num_layers, index=0)
        self.specialExtractor = TAB_DIFF(args, num_layer=num_layers, index=1)
        args.dilation = num_layers[0] + num_layers[1]
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
        self.merge = nn.Sequential(
            nn.Linear(args.dilation, 1),
            nn.Dropout(0.2)
        )
        # self.classifier = nn.Sequential(
        #     Rearrange('N L C  -> N C L'),
        #     nn.AdaptiveAvgPool1d(1),
        #     Rearrange('N C L -> N (C L)'),
        #     nn.Linear(in_features=args.filters * args.dilation, out_features=args.filters),
        #     nn.Linear(in_features=args.filters, out_features=args.num_class)
        # )
        self.classifier = nn.Sequential(
            Rearrange("N C L H -> N C (L H)"),
            nn.AdaptiveAvgPool1d(1),
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
        x_f = self.prepare(x)
        x_b = self.prepare(torch.flip(x, dims=[-1]))
        x_1, x, _ = self.generalExtractor(x_f, x_b, mask)
        x_2, x = self.specialExtractor(x)

        # for ADD_BiDIFF ADD_ADD, ADD_Conv, ADD_AT
        # x_f = self.prepare(x)
        # x_b = self.prepare(torch.flip(x, dims=[-1]))
        # x_1, x_f, x_b = self.generalExtractor(x_f, x_b)
        # x_2, _, _ = self.specialExtractor(x_f, x_b)

        x = self.merge(torch.cat([x_1, x_2], dim=-1))
        # x = torch.mean(torch.cat([x_1, x_2], dim=-1), dim=-1, keepdim=True)   # 另一种平均池化的方法
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    args = Args()
    seed_everything(34)
    x = torch.rand([16, 39, 313]).cuda()
    model = Transformer_DeltaTIM(args).cuda()
    y1 = model(x, index=0)
    y2 = model(x, index=1)
    print(torch.sum(torch.abs(y1 - y2)))
    pass
