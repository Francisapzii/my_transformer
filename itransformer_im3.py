import torch
from torch import nn
from embedding import Embedding


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=0.001):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        avg = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x = (x - avg) / torch.sqrt(var + self.eps)
        x = self.gamma * x + self.beta
        return x


class Attention(nn.Module):
    def __init__(self, d_model, head_num, dropout):
        super(Attention, self).__init__()
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.head_num = head_num
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = LayerNorm(d_model)

    def forward(self, q, k, v, mask):
        print("the very start, v.shape: ",v.shape)
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        seq_num, max_seq_len_q, d_model = q.size()
        head_num, d_tensor = self.head_num, d_model // self.head_num
        q = q.view(seq_num, max_seq_len_q, head_num, d_tensor)
        q = q.transpose(1, 2)
        seq_num, max_seq_len, d_model = k.size()
        k = k.view(seq_num, max_seq_len, head_num, d_tensor).transpose(1, 2)
        v = v.view(seq_num, max_seq_len, head_num, d_tensor).transpose(1, 2)
        alpha = q @ k.transpose(2, 3)
        if mask is not None:
            alpha = alpha.masked_fill(mask, -100000)
        score = self.softmax(alpha)
        x = score @ v
        x = x.transpose(1, 2).contiguous().view(seq_num, max_seq_len_q, d_model)
        x = self.linear(x)
        print("输入x经过attention处理后，其形状不变：（seq_num, max_seq_len(Q), d_model）", x.shape)
        return x


print("attention test || " * 8)
x = torch.randn(3, 6, 36)
src_mask = torch.tensor([[False, False, False, False, True, True],
                         [False, False, False, True, True, True],
                         [False, False, False, False, False, False]])
src_mask = src_mask.unsqueeze(1).unsqueeze(1)
att = Attention(36, 4, 0.2)
x = att(x, x, x, src_mask)
print("shape after attention: ", x.shape)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, head_num, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = Attention(d_model, head_num, dropout)
        self.dropout0 = nn.Dropout(p=dropout)
        self.norm0 = LayerNorm(d_model)
        self.up = nn.Linear(d_model, 4*d_model)
        self.relu0 = nn.ReLU()
        self.down = nn.Linear(4*d_model, d_model)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)
        self.norm1 = LayerNorm(d_model)

    def forward(self, x, mask):
        x_ = x
        x = self.attention(x, x, x, mask)
        x = self.dropout0(x)
        x = self.norm0(x + x_)
        x_ = x
        x = self.up(x)
        x = self.relu0(x)
        x = self.down(x)
        x = self.dropout1(x)
        x = self.norm1(x + x_)
        return x


print("EncoderLayer TEST || " * 8)
enc_layer = EncoderLayer(d_model=36, head_num=3, dropout=0.2)
x = enc_layer(x, src_mask)
print("shape after encoder layer: ", x.shape)


class Encoder(nn.Module):
    def __init__(self, d_model, head_num, dropout):
        super(Encoder, self).__init__()
        self.embedding = Embedding(5000, 36, 2000)
        self.encoders = nn.ModuleList(EncoderLayer(d_model, head_num, dropout) for _ in range(5))

    def forward(self, x, mask):
        x = self.embedding(x)
        print("x.shape after embedding: ", x.shape)
        for layer in self.encoders:
            x = layer(x, mask)
        return x


print("Encoder TEST || " * 8)
x = torch.randint(0, 3000, size=(3, 6))
src_mask = torch.tensor([[False, False, False, False, True, True],
                         [False, False, False, True, True, True],
                         [False, False, False, False, False, False]])
src_mask = src_mask.unsqueeze(1).unsqueeze(1)
enc = Encoder(36, 6, 0.2)
enc_out = enc(x, src_mask)
print("shape after encoder (with embedding):", enc_out.shape)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, head_num, dropout):
        super(DecoderLayer, self).__init__()
        self.attention = Attention(d_model, head_num, dropout)
        self.dropout0 = nn.Dropout(p=dropout)
        self.norm0 = LayerNorm(d_model)
        self.dec_enc_attention = Attention(d_model, head_num, dropout)
        self.dropout1 = nn.Dropout(p=dropout)
        self.norm1 = LayerNorm(d_model)
        self.up = nn.Linear(d_model, 4*d_model)
        self.relu0 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout)
        self.down = nn.Linear(4*d_model, d_model)
        self.relu1 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=dropout)
        self.norm2 = LayerNorm(d_model)

    def forward(self, dec, enc, src_mask, tgt_mask):
        x_ = dec
        x = self.attention(dec, dec, dec, tgt_mask)
        x = self.dropout0(x)
        x = self.norm0(x + x_)
        if enc is not None:
            x_ = x
            x = self.dec_enc_attention(x, enc, enc, src_mask)
            x = self.dropout1(x)
            x = self.norm1(x + x_)
        x_ = x
        x = self.up(x)
        x = self.relu0(x)
        x = self.dropout2(x)
        x = self.down(x)
        x = self.dropout3(x)
        x = self.norm2(x + x_)
        return x


print("Decoder Layer TEST || " * 8)
dec_layer = DecoderLayer(36, 6, 0.2)
y = torch.randn(3, 10, 36)
padding_mask = torch.tensor([[False, False, False, False, False, False, False, False, True, True],
                             [False, False, False, False, False, False, True, True, True, True],
                             [False, False, False, False, False, True, True, True, True, True]])
padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
look_ahead_mask = torch.tril(torch.ones(10, 10)) == 0
look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(0)
tgt_mask = padding_mask | look_ahead_mask
print("target mask 的形状：", tgt_mask.shape)
y = dec_layer(y, enc_out, src_mask, tgt_mask)
print("shape after decoder: ", y.shape)
print("target mask 的形状：", tgt_mask.shape)


class Decoder(nn.Module):
    def __init__(self, d_model, head_num, dropout, voc_size=17):
        super(Decoder, self).__init__()
        self.embedding = Embedding(5000, 36, 2000)
        self.decoders = nn.ModuleList(DecoderLayer(d_model, head_num, dropout) for _ in range(6))
        self.to_word = nn.Linear(d_model, voc_size)

    def forward(self, dec, enc, src_mask, tgt_mask):
        x = self.embedding(dec)
        for layer in self.decoders:
            x = layer(x, enc, src_mask, tgt_mask)

        x = self.to_word(x)
        return x


print("Decoder TEST || " * 8)
y = torch.randint(0, 3000, size=(3, 10))
decoder = Decoder(36, 6, 0.2)
z = decoder(y, enc_out, src_mask, tgt_mask)
print("shape after decoder: ", z.shape)


class Transformer(nn.Module):
    def __init__(self, d_model, head_num, dropout):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model, head_num, dropout)
        self.decoder = Decoder(d_model, head_num, dropout)

    def forward(self, enc, dec, src_mask, tgt_mask):
        x = self.encoder(enc, src_mask)
        x = self.decoder(dec, x, src_mask, tgt_mask)
        return x


print("Transformer TEST || " * 8)
trans = Transformer(36, 6, 0.2)
u = trans(x, y, src_mask, tgt_mask)
print("shape after transformer: ", u.shape)
