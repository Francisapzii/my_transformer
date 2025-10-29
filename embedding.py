import torch
from torch import nn


class TokenEmbedding(nn.Embedding):
    def __init__(self, voc_size, d_model):
        super(TokenEmbedding, self).__init__(voc_size, d_model, padding_idx=1)


tb = TokenEmbedding(5000, 36)
x = torch.randint(0, 5000, size=(3, 9))
u = tb(x)
print(u.shape)
print(u[0, 0, 1])
# y = torch.randint(5000, 10000, size= (1, 5))
# v = tb(y)
# print(v.shape)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]


pe = PositionalEncoding(36, 20000)
z = pe(x)
print(z.shape)
print(z[0])


class Embedding(nn.Module):
    def __init__(self, voc_size, d_model, max_len):
        super(Embedding, self).__init__()
        self.max_len = max_len
        self.token_embedding = TokenEmbedding(voc_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

    def forward(self, x):
        token_embedding = self.token_embedding(x)
        pos_encoding = self.positional_encoding(x)
        return token_embedding + pos_encoding


eb = Embedding(5000, 36, 20000)
a = eb(x)
print(a.shape)
print(a[0, 0])