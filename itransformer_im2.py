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
        print("在encoder中，q, k, v的形状相同：(seq_num_in_one_batch, max_seq_len, d_model)")
        print("在decoder中，k、v的形状完全相同，k，v来自于encoder，所以，与encoder中的k或v的形状，也是完全想同。")
        print("在decoder中，q的形状分两种情况: ")
        print(
            "1)推理时，第一次推理输入是start，然后输出一个词(A)，这个词与start组成序列（start,A）再次输入encoder，输出B，再次组合成序列（start,A,B），并再次输入到encoder")
        print(
            "  因此，在推理时，decoder的q的形状是变化的, 当推理出N个词时，q的形状为：(seq_num_in_one_batch, N+1, d_model)")
        print(
            "  因此，可以得出结论，decoder中q的形状，在推理阶段，与encoder中的输入数据的形状，仅仅在seq_num_in_one_batch和d_model是相同的")
        print("   即：每个token的向量长度相同(d_model)，以及，每个batch中的有多少个序列（seq_num_in_one_batch）是相同的")
        print(
            "2)训练时，target是已知的，依据与encoder输入的batch中对应的target batch，q的形状为: (seq_num_in_one_batch, max_seq_len, d_model)")
        print("   训练阶段decoder的q的形状，与encoder的输入数据，在seq_num_in_one_batch和d_model也是相同的")
        print("   encoder的max_seq_len, 是输入数据中的一个批量中的多个序列中，最长的那个序列的长度。")
        print("   训练阶段decoder的max_seq_len, 是目标数据中的一个批量中的多个序列中，最长的那个序列的长度。")
        print("\n通过形状变换，将q，k，v转换成多头：(seq_num_in_one_batch, head_num, max_seq_len, d_model/head_num)")
        print("首先，变换成：(seq_num_in_one_batch, max_seq_len, head_num, d_model/head_num)")
        seq_num, max_seq_len_q, d_model = q.size()
        head_num, d_tensor = self.head_num, d_model // self.head_num
        q = q.view(seq_num, max_seq_len_q, head_num, d_tensor)
        print("然后，转换维度1和2,变换成：(seq_num_in_one_batch, head_num, max_seq_len, d_model/head_num)")
        q = q.transpose(1, 2)
        print("对q和v执行同样操作，q与v形状相同")
        seq_num, max_seq_len, d_model = k.size()
        k = k.view(seq_num, max_seq_len, head_num, d_tensor).transpose(1, 2)
        v = v.view(seq_num, max_seq_len, head_num, d_tensor).transpose(1, 2)
        print("用每个token的q去查询每个token的k，得到这个token与其他token的相关性；alpha = Q @ K.transpose(2,3)")
        alpha = q @ k.transpose(2, 3)
        print("alpha的形状：", alpha.shape)
        print("\n这个查询，需要屏蔽掉部分内容：")
        print("1) 当q在encoder中时，source_mask将会把短序列中无用token标记出来")
        print("   例如：一个batch的输入序列[[1,2,pad,pad],[2,3,4,5]]")
        print("   第一个序列只有两个有效token，其他用pad填充，以便补齐。这样source_mask就是[[0,0,1,1],[0,0,0,0]]")
        print(
            "2) 当q在decoder中的第一个attention的推理输入是target序列，其mask通过一个批量中target序列得出（与1)同理），名为padding mask")
        print("   在训练阶段，同理也需要padding mask，然而，还需要look ahead mask")
        print("   look ahead mask用一条线将矩阵分成右上角和左下角；右上角全部为True，标记为需要mask掉的部分。")
        print("   当一个序列的一个q与其他K计算相关性后，通过这个look ahead mask将不需要的计算结果屏蔽掉")
        print("       例如：序列第一个token计算与所有其他token的相关性，得到一行，然而，第一个token并不应该知道其他token")
        print(
            "       所以，这时，look ahead mask的第一行，类似[False, True, True,...]，将相关性得分的第一行的第一个元素之外的其他元素，用一个非常小 值替代。")
        print("       对于第二个查询token，只需要查询第一个与其本身的相关性，MASK是[False, False, True,...]，以此类推，")
        print("   当这个序列比较短时，大于这个序列长度的长须是无效的，这时采用padding mask，将无效token的查询，都设置为一个最小值")
        print("   因此，训练阶段decoder第一个attention的mask，是padding_mask | look_ahead_mask")
        print("        推理阶段decoder第一个attention的mask为空")
        print(
            "3) 当q在decoder中的第二个attention中时，无论推理还是阶段都使用source_mask，因为k和v来自经过encoder处理的输入数据。")
        if mask is not None:
            alpha = alpha.masked_fill(mask, -100000)
        score = self.softmax(alpha)
        print("\nscore的形状：(seq_num, head_num, max_seq_len(Q), max_seq_len(K))", score.shape)
        print("max_seq_len(Q)是查询批量中最长序列的长度")
        print("max_seq_len(K)是输入批量中最长序列的长度")
        print("在encoder中，Q K V的max_seq_len是相等的")
        print("在decoder中，max_seq_len(K)是输入数据的批量中最长序列的长度；max_seq_len(Q)是输出目标批量中最长序列的长度；")
        x = score @ v
        print("score @ v的结果的形状是：(seq_num, head_num, max_seq_len(Q), d_tensor)", x.shape)
        print("\n下面将结果的形状转换回（seq_num, max_seq_len(Q), d_model）")
        x = x.transpose(1, 2).contiguous().view(seq_num, max_seq_len_q, d_model)
        x = self.linear(x)
        print("输入x经过attention处理后，其形状不变：（seq_num, max_seq_len(Q), d_model）", x.shape)
        return x


# x = torch.randn(3, 6, 36)
# src_mask = torch.tensor([[False, False, False, False, True, True],
#                          [False, False, False, True, True, True],
#                          [False, False, False, False, False, False]])
# src_mask = src_mask.unsqueeze(1).unsqueeze(1)
# att = Attention(36, 4, 0.2)
# x = att(x, x, x, src_mask)


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


# enc_layer = EncoderLayer(d_model=36, head_num=3, dropout=0.2)
# x = enc_layer(x, src_mask)


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


x = torch.randint(0, 3000, size=(3, 6))
src_mask = torch.tensor([[False, False, False, False, True, True],
                         [False, False, False, True, True, True],
                         [False, False, False, False, False, False]])
src_mask = src_mask.unsqueeze(1).unsqueeze(1)
enc = Encoder(36, 6, 0.2)
out = enc(x, src_mask)
print("shape after encoder (with embedding):", out.shape)


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


# dec_layer = DecoderLayer(36, 6, 0.2)
# y = torch.randn(3, 10, 36)
padding_mask = torch.tensor([[False, False, False, False, False, False, False, False, True, True],
                             [False, False, False, False, False, False, True, True, True, True],
                             [False, False, False, False, False, True, True, True, True, True]])
padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
look_ahead_mask = torch.tril(torch.ones(10, 10)) == 0
look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(0)
tgt_mask = padding_mask | look_ahead_mask
print("target mask 的形状：", tgt_mask.shape)
# y = dec_layer(y, x, src_mask, tgt_mask)
# print("shape after decoder: ", y.shape)
# print("target mask 的形状：", tgt_mask.shape)


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


y = torch.randint(0, 3000, size=(3, 10))
# decoder = Decoder(36, 6, 0.2)
# z = decoder(y, x, src_mask, tgt_mask)
# print("shape after decoder: ", z.shape)


class Transformer(nn.Module):
    def __init__(self, d_model, head_num, dropout):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model, head_num, dropout)
        self.decoder = Decoder(d_model, head_num, dropout)

    def forward(self, enc, dec, src_mask, tgt_mask):
        x = self.encoder(enc, src_mask)
        x = self.decoder(dec, x, src_mask, tgt_mask)
        return x


trans = Transformer(36, 6, 0.2)
u = trans(x, y, src_mask, tgt_mask)
print("shape after transformer: ", u.shape)
