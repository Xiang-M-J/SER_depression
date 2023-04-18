import torch.nn as nn
import torch
import numpy as np
from einops.layers.torch import Rearrange
import torch.nn.functional as F

from config import Args


class DotScore(nn.Module):
    def __init__(self, d_model):
        super(DotScore, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.d_model = d_model

    def forward(self, q, k, v, attn_mask):
        """
        attn[i] = Σ(θ[i,j]*V[j]) V[j]代表一段语句中一个单词的值 \n
        θ[i,j] = softmax((Q[i]*(K[j]^T))/(sqrt(d_model))) Q[i]，K[j]也都代表一段语句中一个单词的查询与键 \n
        除以sqrt(d_model)（经验值）是为了防止维数过高时QK^T的值过大导致softmax函数反向传播时发生梯度消失。
        :param q: [batch_size, n_head, seq_len, d_q]
        :param k: [batch_size, n_head, seq_len, d_k]
        :param v: [batch_size, n_head, seq_len, d_v]
        :param attn_mask: [batch_size, n_head, seq_len, seq_len]
        :return score: [batch_size, n_head, seq_len, d_v] 注意力分布乘V得到的分数
        :return attn: [batch_size, n_head, seq_len, seq_len] 注意力分布
        """
        qk = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_model)  # Matmul, Scale
        # qk shape: [batch_size, n_head, seq_len, seq_len]
        # if attn_mask is not None:
        #     qk.masked_fill_(attn_mask, 1e-9)  # optional
        attn = self.softmax(qk)
        score = torch.matmul(attn, v)
        return score, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, d_q, d_k, d_v):
        """
        input shape: [batch_size, seq_len, feature_dim]
        :param d_model:
        :param n_head:
        :param d_q:
        :param d_k:
        :param d_v:
        """
        super(MultiHeadAttention, self).__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model can not be divided by n_head")
        self.Wq = nn.Linear(in_features=d_model, out_features=n_head * d_q, bias=False)  # 生成q（查询向量）的矩阵
        self.Wk = nn.Linear(in_features=d_model, out_features=n_head * d_k, bias=False)
        self.Wv = nn.Linear(in_features=d_model, out_features=n_head * d_v, bias=False)
        self.fc = nn.Linear(in_features=d_v * n_head, out_features=d_model, bias=False)
        self.n_head = n_head
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        self.attention = DotScore(d_model)
        self.layer = nn.LayerNorm(d_model)

    def forward(self, x_q, x_k, x_v, attn_mask):
        """
        :param attn_mask: [batch_size, seq_len, seq_len]
        :param x_q: [batch_size, seq_len, feature_dim]
        :param x_k: [batch_size, seq_len, feature_dim]
        :param x_v: [batch_size, seq_len, feature_dim]
        :return score
        :return attn
        """
        batch_size = x_q.shape[0]
        identity_q = x_q  # 在解码器中的multiHeadAttention，残差连接是q，从编码器中传入的是k和v，
        # 而在自注意力中，由于x_q, x_k, x_v相同，所以直接选择x_q作为作为残差连接比较合适
        q = self.Wq(x_q).view(batch_size, -1, self.n_head, self.d_q).transpose(1, 2)
        # transpose(1,2)将一个head对应的一个句子的q向量放在最后，方便计算
        # shape: [batch_size, n_head, seq_len, d_q]
        k = self.Wk(x_k).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        # shape: [batch_size, n_head, seq_len, d_k]
        v = self.Wv(x_v).view(batch_size, -1, self.n_head, self.d_v).transpose(1, 2)
        # shape: [batch_size, n_head, seq_len, d_v]
        score, attn = self.attention(q, k, v, attn_mask)  # score: [batch_size, n_head, seq_len, d_v]
        score = score.transpose(1, 2).reshape(score.shape[0], -1, self.n_head * self.d_v)
        # 把注意力连接起来，一个时间步对应的n_head个d_v维的分数，将n_head个分数堆起来得到n_head*d_v维的分数，
        # 之后该分数需要和权重矩阵W相乘，得到d_model维的加权分数(后面还要与残差相加，维度需要保持一致，所以变成d_model维)，
        # 由于一共有seq_len个时间步，所以最后score被重整成[batch_size, seq_len, n_head*d_v] 乘以[n_head*d_v, d_model]全连接层
        # 得到[batch_size, seq_len, d_model]，后面经过残差连接和layerNorm作为输入进feed forward层
        score = self.fc(score)
        score = self.layer(score + identity_q)
        return score, attn


class AFTLocalAttention(nn.Module):
    def __init__(self, d_model: int, seq_len: int, local_window_size: int) -> None:
        super(AFTLocalAttention, self).__init__()
        self.Wq = nn.Linear(d_model, d_model, bias=True)
        self.Wk = nn.Linear(d_model, d_model, bias=True)
        self.Wv = nn.Linear(d_model, d_model, bias=True)
        self.seq_len = seq_len
        self.local_window_size = local_window_size
        self.w_bias = nn.Parameter(torch.Tensor(seq_len, seq_len))
        nn.init.xavier_uniform_(self.w_bias)
        self.local_mask = nn.Parameter(data=self.get_local_mask(), requires_grad=False)
        self.ACT = nn.Sigmoid()
        self.output = nn.Linear(d_model, d_model)

    def get_local_mask(self):
        # Initialize to ones
        local_mask = torch.ones(self.seq_len, self.seq_len, dtype=torch.bool)
        # Make t' - t >= s zero
        local_mask = torch.tril(local_mask, self.local_window_size - 1)
        # Make t - t'>= s zero
        local_mask = torch.triu(local_mask, -(self.local_window_size - 1))
        return local_mask

    def forward(self, q, k, v, attn_mask=None):
        """
        Args:
            q: [batch_size, seq_len, d_model]
            k: [batch_size, seq_len, d_model]
            v: [batch_size, seq_len, d_model]
            mask:
        Returns:
        """
        _, seq_len, _ = q.shape
        query = self.Wq(q)
        key = self.Wk(k)
        value = self.Wv(v)
        w_bias = self.w_bias[:seq_len, :seq_len] * self.local_mask[:seq_len, :seq_len]
        w_bias = w_bias.unsqueeze(0)
        Q_ = self.ACT(query)
        num = torch.exp(w_bias) @ torch.mul(torch.exp(key), value)
        den = (torch.exp(w_bias) @ torch.exp(key))
        num = torch.mul(Q_, num)
        x = num / den
        return self.output(x), num


class AFTSimpleAttention(nn.Module):
    def __init__(self, d_model: int) -> None:
        super(AFTSimpleAttention, self).__init__()
        self.Wq = nn.Linear(d_model, d_model, bias=True)
        self.Wk = nn.Linear(d_model, d_model, bias=True)
        self.Wv = nn.Linear(d_model, d_model, bias=True)
        self.ACT = nn.Sigmoid()
        self.output = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, attn_mask=None):
        """
        Args:
            q: [batch_size, seq_len, d_model]
            k: [batch_size, seq_len, d_model]
            v: [batch_size, seq_len, d_model]
            mask:
        Returns:
        """
        _, seq_len, _ = q.shape
        query = self.Wq(q)
        key = self.Wk(k)
        value = self.Wv(v)
        Q_ = self.ACT(query)
        num = torch.mul(torch.exp(key), value)
        den = torch.exp(key)
        num = torch.mul(Q_, num)
        x = num / den
        return self.output(x), num


class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding.
    This concept was proposed in the "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: query, key, value, pos_embedding, mask
        - **query** (batch, time, dim): Tensor containing query vector
        - **key** (batch, time, dim): Tensor containing key vector
        - **value** (batch, time, dim): Tensor containing value vector
        - **pos_embedding** (batch, time, dim): Positional embedding tensor
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

    Returns:
        - **outputs**: Tensor produces by relative multi head attention module.
    """

    def __init__(self, d_model: int = 512, num_heads: int = 16, dropout_p: float = 0.1):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = np.sqrt(d_model)

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)
        self.softmax = nn.Softmax(dim=-1)
        self.out_proj = nn.Linear(d_model, d_model)

    @staticmethod
    def compute_relative_positional_encoding(pos_score):
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

        return pos_score

    def forward(self, query, key, value, pos_embedding, attn_mask=None):
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        # [batch, time, n_head, d_head]
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        # [batch, n_head, time, d_head]
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        # [batch, n_head, time, d_head]
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)
        # [batch, time, n_head, d_head]

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))  # 论文中的 a+c 项
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))  # 论文中的 b+d 项
        pos_score = self.compute_relative_positional_encoding(pos_score)  # 相对注意力位置编码

        score = (content_score + pos_score) / self.sqrt_dim

        # if attn_mask is not None:
        #     attn_mask = attn_mask.unsqueeze(1)
        #     score.masked_fill_(attn_mask, -1e9)
        attn = self.softmax(score)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context), attn


class TIM_Attention(nn.Module):
    """
    MultiHead
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
        # self.norm2 = nn.LayerNorm(args.filters )
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
        return score  # shape: [N L C]


class TIM_Attention_AF(nn.Module):
    """
    Attention Free
    """

    def __init__(self, args: Args) -> None:
        super(TIM_Attention_AF, self).__init__()
        self.Wq = nn.Sequential(
            Rearrange("N C L H -> N L H C"),
            nn.Linear(in_features=args.filters, out_features=args.d_qkv),
            Rearrange("N L H C -> N L (H C)"),
            # nn.Conv2d(in_channels=args.filters, out_channels=args.d_qkv, kernel_size=1, padding="same"),
            # Rearrange("N C L H -> N L (H C)")
            # [batch_size, feature_dim, seq_len, n_head] -> [batch_size, n_head, seq_len, feature_dim ]
        )
        self.Wk = nn.Sequential(
            Rearrange("N C L H -> N L H C"),
            nn.Linear(in_features=args.filters, out_features=args.d_qkv),
            Rearrange("N L H C -> N L (H C)"),
            # nn.Conv2d(in_channels=args.filters, out_channels=args.d_qkv, kernel_size=1, padding="same"),
            # Rearrange("N C L H -> N L (H C)")
        )
        self.Wv = nn.Sequential(
            Rearrange("N C L H -> N L H C"),
            nn.Linear(in_features=args.filters, out_features=args.d_qkv),
            Rearrange("N L H C -> N L (H C)"),
            # nn.Conv2d(in_channels=args.filters, out_channels=args.d_qkv, kernel_size=1, padding="same"),
            # Rearrange("N C L H -> N L (H C)")
        )

        self.dropout = nn.Dropout(0.2)
        self.x_flatten = Rearrange("N C L H -> N L (H C)")
        # self.layer_norm = nn.LayerNorm([args.filters, args.seq_len, args.dilation])
        self.norm = nn.LayerNorm(args.filters * args.dilation)
        self.seq_len = args.seq_len
        self.w_bias = nn.Parameter(torch.Tensor(self.seq_len, self.seq_len))
        nn.init.xavier_uniform_(self.w_bias)
        self.local_mask = nn.Parameter(data=self.get_local_mask(local_window_size=64),
                                       requires_grad=False)
        self.ACT = nn.Sigmoid()
        self.output = nn.Sequential(
            nn.Linear(args.d_qkv * args.dilation, args.d_ff),
            nn.Linear(args.d_ff, args.dilation * args.filters)
        )

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
        # num = torch.exp(w_bias) @ torch.mul(torch.exp(key), value)

        num = torch.mul(torch.exp(key), value)
        # den = (torch.exp(w_bias) @ torch.exp(key))
        den = torch.exp(key)
        num = torch.mul(Q_, num)
        num = self.dropout(num)
        score = self.norm(self.output(num / den) + self.x_flatten(x))
        # score = self.norm(self.fc(score) + score)
        return score  # [N L (H C)]


class TIM_Attention_SE(nn.Module):
    """
    通道注意力(SENet)
    """

    def __init__(self, arg: Args):
        super(TIM_Attention_SE, self).__init__()
        self.pool1 = nn.Sequential(
            Rearrange("N C L H -> N H C L"),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(
            nn.Conv2d(arg.dilation, arg.dilation, kernel_size=1),
            # nn.ReLU(),
            # nn.Conv2d(arg.dilation, arg.dilation, kernel_size=1),
            # nn.ReLU(),
            # nn.Conv2d(arg.dilation, arg.dilation, kernel_size=1),
            nn.Sigmoid(),
            Rearrange("N H C L -> N C L H")
        )
        self.output = nn.Sequential(
            Rearrange("N C L H -> N L (H C)")
        )

    def forward(self, x, mask=None):
        attn1 = self.pool1(x)
        attn = self.fc(attn1)
        x = x * attn
        return self.output(x)


class TIM_Attention_CS(nn.Module):
    """

    """

    def __init__(self, arg: Args):
        super(TIM_Attention_CS, self).__init__()
        self.in_proj = Rearrange("N C L H -> N H C L")
        self.channel = nn.Sequential(
            nn.Conv2d(arg.dilation, arg.dilation, kernel_size=1),
            # nn.ReLU(),
            nn.Conv2d(arg.dilation, arg.dilation, kernel_size=1),
            # nn.ReLU(),
            # nn.Conv2d(arg.dilation, arg.dilation, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            # nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.out_proj = nn.Sequential(
            Rearrange("N H C L -> N L (H C)")
        )

    def forward(self, x, mask=None):
        x = self.in_proj(x)
        attn = self.channel(F.adaptive_avg_pool2d(x, (1, 1))) + self.channel(F.adaptive_max_pool2d(x, (1, 1)))
        x = x * attn
        x_compress = torch.cat([torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)], dim=1)
        attn = self.spatial(x_compress)
        x = x * attn
        return self.out_proj(x)


def get_attention(args: Args):
    """
    返回注意力模块
    """
    if args.attention_type == "MH":
        print("Multi-head")
        return TIM_Attention(args)
    elif args.attention_type == 'AF':
        print("Attention free local")
        return TIM_Attention_AF(args)
    elif args.attention_type == "SE":
        print("SE ")
        return TIM_Attention_SE(args)
    elif args.attention_type == "CS":
        print("CS ")
        return TIM_Attention_CS(args)
    else:
        raise NotImplementedError
