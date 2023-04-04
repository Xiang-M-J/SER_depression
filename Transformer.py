import torch
import torch.nn as nn
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PositionEncoding(nn.Module):
    """
    input:[batch_size, seq_len, feature_dim]
    """

    def __init__(self, d_model, max_len=5000, p=0):
        super(PositionEncoding, self).__init__()
        pe = np.array([[pos / (10000 ** (2 * i / d_model)) for i in range(d_model)] for pos in range(max_len)])
        pe[1:, ::2] = np.sin(pe[1:, ::2])  # 给每个词向量加入位置编码
        pe[1:, ::2] = np.cos(pe[1:, ::2])
        pe = torch.FloatTensor(pe)
        self.d_model = d_model
        self.pe = pe.to(device)
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        seq_len = x.shape[1]
        if x.shape[-1] != self.d_model:
            raise ValueError(f'input feature_dim is {x.shape[-1]}, do not match {self.d_model}')
        x = x + self.pe[:seq_len, :]
        x = self.dropout(x)
        return x

    def get_pos_embedding(self, x):
        seq_len = x.shape[1]
        return self.pe[:seq_len, :]


def get_padding_mask(mask, n_head: int, tgt_seq_len=None):
    """
    :param tgt_seq_len:
    :param n_head:
    :param mask:
    :return: attn_mask(bool): [batch_size, n_head, seq_len, seq_len] False代表未遮蔽，True代表遮蔽
    """
    batch_size, seq_len = mask.shape
    attn_mask = (mask == 0).unsqueeze(1)
    if tgt_seq_len is None:
        attn_mask = attn_mask.expand(batch_size, seq_len, seq_len).unsqueeze(1)
        attn_mask = attn_mask.expand(batch_size, n_head, seq_len, seq_len)
    else:
        attn_mask = attn_mask.expand(batch_size, tgt_seq_len, seq_len).unsqueeze(1)
        attn_mask = attn_mask.expand(batch_size, n_head, tgt_seq_len, seq_len)
    return attn_mask.to(device)


def get_seq_mask(batch_size, seq_len, n_head):
    """
    获得解码器输入的mask，防止泄漏信息（上三角mask）
    :param n_head:
    :param batch_size:
    :param seq_len:
    :return: mask[batch_size, n_head, seq_len, seq_len] 1代表遮蔽，0代表不遮蔽
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.unsqueeze(0)
    mask = mask.expand(batch_size, seq_len, seq_len)
    mask = mask.unsqueeze(1)
    mask = mask.expand(batch_size, n_head, seq_len, seq_len)
    return mask.to(device)


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
        if attn_mask is not None:
            qk.masked_fill_(attn_mask, 1e-9)  # optional
        attn = self.softmax(qk)
        score = torch.matmul(attn, v)
        return score, attn


class feedForward(nn.Module):
    def __init__(self, d_model, dim_feed_forward):
        """
        前馈层
        :param d_model: 词编码维度
        :param dim_feed_forward: 前馈层深度
        """
        super(feedForward, self).__init__()
        self.layer = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(in_features=d_model, out_features=dim_feed_forward)
        self.fc2 = nn.Linear(in_features=dim_feed_forward, out_features=d_model)
        self.act = nn.ReLU()

    def forward(self, x):
        identity_x = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.layer(x + identity_x)
        return x


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


def get_local_mask(seq_len, local_window_size):
    # Initialize to ones
    local_mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
    # Make t' - t >= s zero
    local_mask = torch.tril(local_mask, local_window_size - 1)
    # Make t - t'>= s zero
    local_mask = torch.triu(local_mask, -(local_window_size - 1))
    return local_mask


class AFTLocalAttention(nn.Module):
    def __init__(self, d_model: int, seq_len: int, local_window_size: int) -> None:
        super(AFTLocalAttention, self).__init__()
        self.Wq = nn.Linear(d_model, d_model, bias=True)
        self.Wk = nn.Linear(d_model, d_model, bias=True)
        self.Wv = nn.Linear(d_model, d_model, bias=True)

        self.w_bias = nn.Parameter(torch.Tensor(seq_len, seq_len))
        nn.init.xavier_uniform_(self.w_bias)
        self.local_mask = nn.Parameter(data=get_local_mask(seq_len=seq_len, local_window_size=local_window_size),
                                       requires_grad=False)
        self.ACT = nn.Sigmoid()
        self.output = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
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

    def forward(self, query, key, value, pos_embedding, mask=None):
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
        pos_score = compute_relative_positional_encoding(pos_score)  # 相对注意力位置编码

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9)
        attn = self.softmax(score)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context), attn


def compute_relative_positional_encoding(pos_score):
    batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
    zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
    padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

    padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
    pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

    return pos_score


class encoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_q, d_k, d_v, dim_feed_forward):
        super(encoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_head, d_q, d_k, d_v)
        # self.attention = AFTLocalAttention(d_model, seq_len=313, local_window_size=256)
        # self.pos = PositionEncoding(d_model)
        # self.attention = RelativeMultiHeadAttention(d_model, n_head)
        self.ffn = feedForward(d_model, dim_feed_forward)

    def forward(self, x, attn_mask=None):
        # score, attn = self.attention(x, x, x, attn_mask)
        # pos_embed = self.pos.get_pos_embedding(x)
        # pos_embed = pos_embed.unsqueeze(0)
        # score, attn = self.attention(x, x, x, pos_embed.repeat(x.shape[0], 1, 1))
        score, attn = self.attention(x, x, x, attn_mask)
        score = self.ffn(score)
        return score, attn


class Encoder(nn.Module):
    def __init__(self, d_model=512, n_head=8, d_q=64, d_k=64, d_v=64, dim_feed_forward=1024, n_layer: int = 4):
        super(Encoder, self).__init__()
        # self.embedding = nn.Embedding(1000, d_model)
        self.pos = PositionEncoding(d_model)
        self.encodeLayers = nn.ModuleList([])
        self.n_head = n_head
        for i in range(n_layer):
            self.encodeLayers.append(encoderLayer(d_model, n_head, d_q, d_k, d_v, dim_feed_forward))

    def forward(self, x, mask=None):
        """
        :param x:
        :param mask:
        :return:
        """
        x = self.pos(x.transpose(1, 2))
        attns = []
        if mask is not None:
            attn_mask = get_padding_mask(mask, n_head=self.n_head)
        else:
            attn_mask = None
        for layer in self.encodeLayers:
            x, attn = layer(x, attn_mask)
            attns.append(attn)
        return x, attns
