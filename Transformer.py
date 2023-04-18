import torch
import torch.nn as nn
import numpy as np
from attention import MultiHeadAttention, RelativeMultiHeadAttention, AFTLocalAttention

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
