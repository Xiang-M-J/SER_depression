import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from CNN import WeightLayer, Temporal_Aware_Block, Temporal_Aware_Block_design, Chomp1d, SpatialDropout
from TransformerEncoder import TransformerEncoder, TransformerEncoderLayer
from config import Args
from utils import seed_everything, mask_input


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

    def forward(self, x, mask=None):
        query = self.Wq(x)
        key = self.Wk(x)
        value = self.Wv(x)
        attn = torch.matmul(query, key.transpose(-1, -2)) / (np.sqrt(self.d_model))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        # if mask is not None:
        #     attn.masked_fill_(mask, 1e-9)  # optional
        score = torch.matmul(attn, value)
        score = self.norm(self.score_flatten(score) + self.x_flatten(x))
        score = self.norm(self.fc(score) + score)
        return score  # shape: [N L H*C]


class AFTLocalAttention(nn.Module):
    def __init__(self, args: Args) -> None:
        super(AFTLocalAttention, self).__init__()
        self.Wq = nn.Sequential(
            nn.Conv2d(in_channels=args.filters, out_channels=args.d_qkv, kernel_size=1, padding="same"),
            Rearrange("N C L H -> N L (H C)")
            # [batch_size, feature_dim, seq_len, n_head] -> [batch_size, n_head, seq_len, feature_dim ]
        )
        self.Wk = nn.Sequential(
            nn.Conv2d(in_channels=args.filters, out_channels=args.d_qkv, kernel_size=1, padding="same"),
            Rearrange("N C L H -> N L (H C)")
        )
        self.Wv = nn.Sequential(
            nn.Conv2d(in_channels=args.filters, out_channels=args.d_qkv, kernel_size=1, padding="same"),
            Rearrange("N C L H -> N L (H C)")
        )

        self.dropout = nn.Dropout(0.1)
        self.x_flatten = Rearrange("N C L H -> N L (H C)")
        # self.layer_norm = nn.LayerNorm([args.filters, args.seq_len, args.dilation])
        self.norm = nn.LayerNorm(args.dilation * args.filters)
        # self.fc = nn.Sequential(
        #     nn.Linear(in_features=args.filters * args.dilation, out_features=args.d_ff),
        #     # nn.Dropout(0.2),
        #     nn.ReLU(),
        #     nn.Linear(in_features=args.d_ff, out_features=args.filters * args.dilation),
        #     nn.ReLU()
        # )
        self.seq_len = args.seq_len
        self.w_bias = nn.Parameter(torch.Tensor(self.seq_len, self.seq_len))
        nn.init.xavier_uniform_(self.w_bias)
        self.local_mask = nn.Parameter(data=self.get_local_mask(local_window_size=128),
                                       requires_grad=False)
        self.ACT = nn.Sigmoid()
        self.output = nn.Linear(args.d_qkv * args.dilation, args.dilation * args.filters)

    def get_local_mask(self, local_window_size):
        # Initialize to ones
        local_mask = torch.ones(self.seq_len, self.seq_len, dtype=torch.bool)
        # Make t' - t >= s zero
        local_mask = torch.tril(local_mask, local_window_size - 1)
        # Make t - t'>= s zero
        local_mask = torch.triu(local_mask, -(local_window_size - 1))
        return local_mask

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, feature_dim, seq_len, n_head]
            mask:
        Returns:
        """
        _, _, seq_len, _ = x.shape
        query = self.Wq(x)
        key = self.Wk(x)
        value = self.Wv(x)
        w_bias = self.w_bias[:seq_len, :seq_len] * self.local_mask[:seq_len, :seq_len]
        w_bias = w_bias.unsqueeze(0)
        Q_ = self.ACT(query)
        num = torch.exp(w_bias) @ torch.mul(torch.exp(key), value)
        den = (torch.exp(w_bias) @ torch.exp(key))
        num = torch.mul(Q_, num)
        score = self.norm(self.output(num / den) + self.x_flatten(x))
        # score = self.norm(self.fc(score) + score)
        return score


def attention(args: Args):
    """
    返回注意力模块
    """
    if args.attention_type == "MH":
        print("Multi-head")
        return TIM_Attention(args)
    elif args.attention_type == 'AF':
        print("Attention free local")
        return AFTLocalAttention(args)
    else:
        raise NotImplementedError


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
                Temporal_Aware_Block(feature_dim=args.filters, filters=args.filters, kernel_size=args.kernel_size,
                                     dilation=i, dropout=args.drop_rate)
            )
        self.drop = nn.Dropout(p=args.drop_rate)

        self.attn = attention(args)
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


class AT_TIM_v2(nn.Module):
    """
    基于注意力机制的TIM，用注意力机制取代了weight layer，也可以取代Transformer
    """

    def __init__(self, args: Args):
        super(AT_TIM_v2, self).__init__()
        self.name = "AT_TIM_v2"
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
        self.fusion = nn.Conv1d(in_channels=2 * args.filters, out_channels=args.filters, kernel_size=1, padding="same")
        self.attn = attention(args)
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
        skip_stack = []
        skip_out_f = x_f
        skip_out_b = x_b
        for layer in self.dilation_layer:
            skip_out_f = layer(skip_out_f)
            skip_out_b = layer(skip_out_b)
            tmp = self.fusion(torch.cat([skip_out_f, skip_out_b], dim=1))
            skip_stack.append(tmp)
        skip_stack = torch.stack(skip_stack, dim=-1)
        if mask is not None:
            x = self.attn(skip_stack, mask)
        else:
            x = self.attn(skip_stack)
        x = self.classifier(x)
        return x


class AT_TIM_v3(nn.Module):
    """
    基于注意力机制的TIM，用注意力机制取代了weight layer，也可以取代Transformer
    """

    def __init__(self, args: Args):
        super(AT_TIM_v3, self).__init__()
        self.name = "AT_TIM_v3"
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
        self.attn = attention(args)
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

        x = self.conv(x)
        skip_stack = []

        for layer in self.dilation_layer:
            x = layer(x)
            skip_stack.append(x)
        skip_stack = torch.stack(skip_stack, dim=-1)
        if mask is not None:
            x = self.attn(skip_stack, mask)
        else:
            x = self.attn(skip_stack)
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

        self.attn = attention(args)

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


class AT_DeltaTIM_v2(nn.Module):
    """
    基于注意力的差分TIM v2
    """

    def __init__(self, args: Args):
        super(AT_DeltaTIM_v2, self).__init__()
        self.name = "AT_DeltaTIM_v2"
        self.dilation_layer = nn.ModuleList([])
        self.fusion = nn.ModuleList([])
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
            self.fusion.append(
                nn.Conv1d(in_channels=2 * args.filters, out_channels=args.filters, kernel_size=1, padding="same"))
        self.drop = nn.Dropout(p=args.drop_rate)
        self.attn = attention(args)
        self.classifier = nn.Sequential(
            # nn.AdaptiveAvgPool2d((args.seq_len, 1)),
            # Rearrange("N C L H -> N C (L H)"),
            # nn.AdaptiveAvgPool1d(1),
            nn.Linear(in_features=args.filters * args.dilation, out_features=1),
            nn.Dropout(0.3),
            Rearrange('N C L -> N (C L)'),
            nn.Linear(in_features=args.seq_len, out_features=args.num_class)
        )

    def forward(self, x, mask=None):
        if self.training:
            x = mask_input(x, 0.2)
        x_f = x
        x_b = torch.flip(x, dims=[-1])
        x_f = self.conv(x_f)
        x_b = self.conv(x_b)
        delta_stack = []
        skip_out = None
        now_skip_out_f = x_f
        now_skip_out_b = x_b
        now_skip_out = None
        for i, layer in enumerate(self.dilation_layer):
            now_skip_out_f = layer(now_skip_out_f)
            now_skip_out_b = layer(now_skip_out_b)
            now_skip_out = self.fusion[i](torch.cat([now_skip_out_f, now_skip_out_b], dim=1))
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


class TAB(nn.Module):
    def __init__(self, args: Args, num_layer: int, middle: bool = True):
        super(TAB, self).__init__()
        self.net = nn.ModuleList([])
        for i in [2 ** i for i in range(num_layer)]:
            self.net.append(
                Temporal_Aware_Block(feature_dim=args.filters, filters=args.filters, kernel_size=args.kernel_size,
                                     dilation=i, dropout=args.drop_rate)
            )
        args.dilation = num_layer
        self.attn = attention(args)
        self.proj = nn.Sequential(
            nn.Linear(args.dilation * args.filters, args.filters),
            Rearrange("N L C -> N C L"),
        ) if middle else None

    def forward(self, x, mask=None):
        stack_layer = []
        for layer in self.net:
            x = layer(x)
            stack_layer.append(x)
        stack_layer = torch.stack(stack_layer, dim=-1)
        if mask is not None:
            x = self.attn(stack_layer, mask)
        else:
            x = self.attn(stack_layer)
        if self.proj is not None:
            x = self.proj(x)
        return x


class TAB_ADD(nn.Module):
    def __init__(self, args: Args, num_layer: int, middle: bool = True):
        super(TAB_ADD, self).__init__()
        self.net = nn.ModuleList([])
        for i in [2 ** i for i in range(num_layer)]:
            self.net.append(
                Temporal_Aware_Block(feature_dim=args.filters, filters=args.filters, kernel_size=args.kernel_size,
                                     dilation=i, dropout=args.drop_rate)
            )
        args.dilation = num_layer
        self.attn = attention(args)
        self.proj = nn.Sequential(
            nn.Linear(args.dilation * args.filters, args.filters),
            Rearrange("N L C -> N C L"),
        ) if middle else None

    def forward(self, x_f, x_b, mask=None):
        stack_layer = []
        for layer in self.net:
            x_f = layer(x_f)
            x_b = layer(x_b)
            tmp = torch.add(x_f, x_b)
            stack_layer.append(tmp)
        stack_layer = torch.stack(stack_layer, dim=-1)
        if mask is not None:
            x = self.attn(stack_layer, mask)
        else:
            x = self.attn(stack_layer)
        if self.proj is not None:
            x = self.proj(x)
        return x


class TAB_DIFF(nn.Module):
    def __init__(self, args: Args, num_layer: int, middle: bool = True):
        super(TAB_DIFF, self).__init__()
        self.net = nn.ModuleList([])
        for i in [2 ** i for i in range(num_layer)]:
            self.net.append(
                Temporal_Aware_Block(feature_dim=args.filters, filters=args.filters, kernel_size=args.kernel_size,
                                     dilation=i, dropout=args.drop_rate)
            )
        args.dilation = num_layer
        self.attn = attention(args)
        self.proj = nn.Sequential(
            nn.Linear(args.dilation * args.filters, args.filters),
            Rearrange("N L C -> N C L"),
        ) if middle else None

    def forward(self, x, mask=None):
        stack_layer = []
        last_x = None
        for layer in self.net:
            x = layer(x)
            if last_x is not None:
                stack_layer.append(x - last_x)
            last_x = x
        stack_layer.append(x)
        stack_layer = torch.stack(stack_layer, dim=-1)
        if mask is not None:
            x = self.attn(stack_layer, mask)
        else:
            x = self.attn(stack_layer)
        if self.proj is not None:
            x = self.proj(x)
        return x


class MultiTIM(nn.Module):
    def __init__(self, args: Args):
        super(MultiTIM, self).__init__()
        self.prepare = nn.Sequential(
            nn.Conv1d(in_channels=args.feature_dim, out_channels=args.filters, kernel_size=1, dilation=1, padding=0),
            nn.BatchNorm1d(args.filters),
            nn.ReLU(),
        )
        self.generalExtractor = TAB(args, num_layer=4)
        self.specialExtractor = TAB_DIFF(args, num_layer=4, middle=False)
        self.classifier = self.classifier = nn.Sequential(
            nn.Linear(in_features=args.filters * args.dilation, out_features=1),
            nn.Dropout(0.3),
            Rearrange('N C L -> N (C L)'),
            nn.Linear(in_features=args.seq_len, out_features=args.num_class)
        )

    def forward(self, x, mask=None):
        x = self.prepare(x)
        x = self.generalExtractor(x, mask)
        x = self.specialExtractor(x, mask)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    args = Args()
    seed_everything(34)
    x = torch.rand([16, 39, 313]).cuda()
    model = MultiTIM(args).cuda()
    y1 = model(x)
    # y2 = model(x, index=1)
    # print(torch.sum(torch.abs(y1 - y2)))
    pass
