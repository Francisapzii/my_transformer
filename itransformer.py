# 首先定义神经网络架构
# 输入为一个序列，例如一个问题：你是谁？
# 这个序列先经过分词器变为若干token，例如，变成4个token，token经过embedding层后，每个token变成一个一维向量，例如：1*512
# 这时输入变为4个向量，形状为：4*512
# 进入QKV，每个token的维度进一步变化，假如hidden-size=1024，那么经过W(q,k,v)后，每个tokenQKV的形状是：1*1024
# 使用一个token的Q， 对每个K进行查询，即让Q*K，每个Q*K后的形状是1*1？
# 进行softmax操作：exp（Q*K）/∑exp（Q*K)，其形状仍然是1*1；这是每个token对于查询token的相关性比例。
# 对这个token的4个相关性比例，分别乘以对应token的V值，然后将这些乘机加在一起，得到一个token的输出，形状为：1*1024
# 最终4个token输出的形状是4*1024
# 这个输出，应该是有问题的，因为这种输出无法与输入进行相加。
# 接下来要进行FEC全连接网络：
# 每个token的形状是1*1024
# 可以将hidden-size设为2048，每个token的输出形状是1*2048；
# 然后，再将输出降维到1024，每个token的输出形状为1*1024；
# 这样就可以与输入相加；
# 每次变换都需要进行非线性激活处理。
import math
import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, head_num):
        super(MultiHeadAttention, self).__init__()
        self.head_num = head_num
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        print("Q, K, V 的形状：(batch_size, length, d_model):", q.shape)
        batch_size, length, d_model = q.size()
        d_tensor = d_model // self.head_num
        # 下面注释掉的q k v三行代码，在进行cross attention操作时，将出现问题。
        # 主要时length的问题，当一个批次的多个序列进入encoder之前，会填充pad使之长度相同，这里的长度就是length。
        # 然而，在cross attention时，q的length是target的最大长度，与输入的最大长度，大概率不同。
        # 因此，k和v的length，应有k或者v来确定，（k和v都来自encoder，形状相同）
        # q = q.view(batch_size, length, self.head_num, d_tensor).transpose(1, 2)
        # k = k.view(batch_size, length, self.head_num, d_tensor).transpose(1, 2)
        # v = v.view(batch_size, length, self.head_num, d_tensor).transpose(1, 2)

        # 修改后的代码如下：
        _, k_length, _ = k.size()
        q = q.view(batch_size, length, self.head_num, d_tensor).transpose(1, 2)
        k = k.view(batch_size, k_length, self.head_num, d_tensor).transpose(1, 2)
        v = v.view(batch_size, k_length, self.head_num, d_tensor).transpose(1, 2)

        print("Q, K, V 变换后的形状：(batch_size, length, d_model):", q.shape)
        alpha = q @ k.transpose(2, 3)
        alpha = alpha / math.sqrt(d_tensor)
        print("alpha 的形状：（batch_size, head_num, length, d_tensor）：", alpha.shape)
        if mask is not None:
            print("mask的形状：", mask.shape)
            alpha = alpha.masked_fill(mask, -100000)

        score = self.softmax(alpha)
        print("score的形状：（batch_size, head_num, length, length）:", score.shape)
        out = score @ v
        print("score@v 的形状：（batch_size, head_num, length, d_tensor）", out.shape)
        out = out.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        out = self.out_linear(out)
        print("out的形状：（batch_size, length, d_model）:", out.shape)
        return out


# 假设 d_model为30，序列的最大长度是10，每个batch 4个序列
# 生成 4个序列，每个序列的长度分别是：8， 10， 7， 3
source_pad = -100
one_batch = torch.randn(4, 10, 36)
one_batch[0, 8:, :] = source_pad
one_batch[2, 7:, :] = source_pad
one_batch[3, 3:, :] = source_pad
# 首先,一个批量中的多个序列，最大长度m（m是这个批量的所有序列中最长的序列的长度）；
# 每个序列对应一条mask，例如，如果一个序列的长度是4，批量的最大长度是6，那么，这条mask是：[1,1,1,1,0,0]
# 然后，当attention的头是n时，那么，就把每个序列生成了n个头，每个头的score的形状时m*m，
# 每行的每个值代表元序列对应序号token与其他token相关性的值；
# 继续上面的例子，最大序列长度是6，那么每个head的score的形状是6*6
# 取出其中第1行，例如：[0.1,0.2,0.3,0.4,0.5,0.6],0.1就是第一个token与自己相关性的分值，0.2是第一个token与第2个token相关性的得分，等等。
# 那么，因为这个序列的有效长度是4， 所以，需要将第5行和第6行置为一个很小的值，这就是mask起作用的地方。
# 不经如此，这个序列被映射到n个head，比如n为5，那么，对这个5个head的score，都要执行同样的mask。
# 对于批量中的其他序列，执行同样的操作。
# 对于一个批量的序列，计算其mask的过程如下：
# 1，依据每个序列的有效长度和所有序列的最大长度max_seq_len，计算每个序列的mask；
# 2. 将这个批量的每个序列的mask组合到一起，形成一个二维形状的数据，每行代表一个序列的mask；
# 3. 首先要清楚，attention的head_num是多少，每个序列经过multihead变换后，将生成head_num个Q K V;
#    Q K V的形状，都是m*(d_model/head_num)，Q * K.tranpose()后，生成score
#    每个head的score的形状，都是max_seq_len * max_seq_len。整体形状是：（批量，head_num, max_seq_len, max_seq_len）
#    score的每一行都是对应token对其他token的查询得分；
#    当查询来自decoder，其他token来在encoder，那么每个head的score形状变为max_seq_len_from_decoder * max_seq_len_from_encoder
# 4. 为适应这种形状，需要对第2步得出的mask做如何变换：
#    mask当前的形状是：seq_num * m，每一行是一个序列的mask；(seq_num是一个batch中有多少个序列)
#    maks = mask.unsqueeze(1).repeat(1, head_num, 1).unsqueeze(-1)
#    经过如上处理后的mask的形状是：（seq_num,head_num, m, 1）
src_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]])
src_mask = src_mask.unsqueeze(1)
src_mask = src_mask.repeat(1, 3, 1).unsqueeze(-2)
src_mask = src_mask == 0
mh = MultiHeadAttention(36, 3)
# mh_out = mh(one_batch, one_batch, one_batch)
x = mh(one_batch, one_batch, one_batch, src_mask)
print(x.shape)


class LayerNorm(nn.Module):
    """这层对每个token的特征向量执行归一化处理：计算当前token中特征向量的均值和方差，将特征向量归一化为均值为0 标准差为1 的分布。"""

    def __init__(self, d_model, eps=0.001):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # 求每行的均值
        # row_avg = x.sum(dim=-1, keepdim=True) / x.shape[-1]
        row_avg = x.mean(dim=-1, keepdim=True)
        # 求每行的方差
        # fc = (x_rem * x_rem).sum(dim=-1, keepdim=True)
        # fc = fc / x.shape[-1]
        fc = x.var(-1, unbiased=False, keepdim=True)
        # (x - 均值)/ 方差
        out = (x - row_avg) / torch.sqrt(fc + self.eps)
        out = out * self.gamma + self.beta
        return out


ln = LayerNorm(d_model=36)
x = ln(x + one_batch)
print("shape after layernorm:", x.shape)
_x = x


class Ffn(nn.Module):
    """FFN对每个token的特征向量独立进行非线性变换。FFN层公式：ffn(x)=ReLU(xW1+b1)W2+b2"""

    def __init__(self, d_model, ddf, drop_rate):
        super(Ffn, self).__init__()
        self.up = nn.Linear(d_model, ddf)
        self.down = nn.Linear(ddf, d_model)
        self.drop_rate = drop_rate
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        x = self.up(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.down(x)
        return x


ffn = Ffn(d_model=36, ddf=36 * 4, drop_rate=0.2)
x = ffn(x)
print("shape after ffn: ", x.shape)
ln = LayerNorm(d_model=36)
x = ln(x + _x)
print("shape after ffn and layernorm: ", x.shape)


class EncodeLayer(nn.Module):
    def __init__(self, d_model, head_num, dropout):
        super(EncodeLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, head_num)
        self.ffn = Ffn(d_model, 4 * d_model, dropout)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm0 = LayerNorm(d_model)
        self.layer_norm1 = LayerNorm(d_model)

    def forward(self, x, src_mask):
        x_ = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        x = self.dropout(x)
        x = self.layer_norm0(x + x_)
        x_ = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.layer_norm1(x + x_)
        return x


encoder = EncodeLayer(d_model=36, head_num=3, dropout=0.2)
x = encoder(one_batch, src_mask)
print("shape after encoder: ", x.shape)
_x = x


class Encoder(nn.Module):
    def __init__(self, d_model, head_num, dropout):
        super(Encoder, self).__init__()
        self.embedding = 3
        self.layers = nn.ModuleList(EncodeLayer(d_model=d_model, head_num=head_num, dropout=dropout) for _ in range(10))

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


enc = Encoder(d_model=36, head_num=3, dropout=0.2)
x = enc(one_batch, src_mask)
print("shape after encoder: ", x.shape)
x_ = x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, head_num, dropout):
        super(DecoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, head_num)
        self.dropout0 = nn.Dropout(p=dropout)
        self.layer_norm0 = LayerNorm(d_model)
        self.cross_attention = MultiHeadAttention(d_model, head_num)
        self.dropout1 = nn.Dropout(p=dropout)
        self.layer_norm1 = LayerNorm(d_model)
        self.ffn = Ffn(d_model, 4 * d_model, dropout)
        self.layer_norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, dec, enc, src_mask, target_mask):
        x = self.attention(dec, dec, dec, target_mask)
        x = self.dropout0(x)
        x = self.layer_norm0(x + dec)
        if enc is not None:
            x_ = x
            x = self.cross_attention(x, enc, enc, src_mask)
            x = self.dropout1(x)
            x = self.layer_norm1(x + x_)
        x_ = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.layer_norm2(x + x_)
        return x


# 在上面的encoder的输入为一个batch，这个batch中有4个序列，
# 在decoder上，输入也必须是一个batch，并且batch中必须是4个序列。
# 假定输出序列的最大长度是5，4个序列的输出长度分别是：[2, 5, 3, 4]，据此生成target_padding_mask；
# 然后生成look_ahead_mask; 批量中生成序列的最大长度是5，所以，self attention的score的形状是5*5，
# 那么，score的形状也是5*5，每行代表了序列中对应位置token对其他token的查询得分。
# 因此，look_ahead_mask的形状也是5*5，而且是一个右上三角都是True的矩阵；
# 两个mask中，将每个序列中需要的部分，标记为true，以便后续使用masked_fill，将这些特征置为一个特别小的数，免除干扰。
target_pad = 0
tgt = torch.randn(4, 5, 36)
target_padding_mask = torch.tensor([[False, False, True, True, True],
                                    [False, False, False, False, False],
                                    [False, False, False, True, True],
                                    [False, False, False, False, True]])
target_padding_mask = target_padding_mask.unsqueeze(1)
look_ahead_mask = (torch.tril(torch.ones(5, 5)) == 0)
look_ahead_mask = look_ahead_mask.unsqueeze(0).repeat(4, 1, 1)
print("target padding mask shape and look ahead mask shape:", target_padding_mask.shape, look_ahead_mask.shape)
target_mask = target_padding_mask | look_ahead_mask
target_mask = target_mask.unsqueeze(1)
# print(target_mask)
# print(look_ahead_mask)
print("shape of target_mask: ", target_mask.shape)
# decl = DecoderLayer(d_model=36, head_num=3, dropout=0.2)
# x = decl(tgt, x, src_mask, target_mask)
# print("shape after decoder layer: ", x.shape)


class Decoder(nn.Module):
    def __init__(self, d_model, head_num, dropout, voc_size):
        super(Decoder, self).__init__()
        self.decs = nn.ModuleList(DecoderLayer(d_model, head_num, dropout) for _ in range(5))
        self.to_word = nn.Linear(d_model, voc_size)

    def forward(self,dec, enc, src_mask, target_mask):
        x = dec
        for layer in self.decs:
            x = layer(x, enc, src_mask, target_mask)
        x = self.to_word(x)
        return x


dec = Decoder(36, 3, 0.2, 20)
y = dec(tgt, x, src_mask, target_mask)
print("shape after decoder: ", y.shape)


class Transformer(nn.Module):
    def __init__(self, d_model, head_num, dropout, voc_size):
        super().__init__()
        self.encoder = Encoder(d_model, head_num, dropout)
        self.decoder = Decoder(d_model, head_num, dropout, voc_size)

    def forward(self, enc, dec, src_mask, tgt_mask):
        x = self.encoder(enc, src_mask)
        x = self.decoder(dec, x, src_mask, tgt_mask)
        return x


transformer = Transformer(36, 3, 0.2, 20)
y = transformer(x, tgt, src_mask, target_mask)
print("shape after transformer: ", y.shape)
