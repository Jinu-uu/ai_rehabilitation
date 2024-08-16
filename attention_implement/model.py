import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransFormerModel(nn.Module):
    def __init__(self,):
        super(TransFormerModel,self).__init__()
        # 인코더
        # 디코더
        # 디코더 output linear
        # softmax
    
    def forward(self,):
        pass


class EncoderLayer(nn.Module):
    def __init__(self, ):
        super(EncoderLayer, self).__init__()
        self.embedding_layer = Embedding().fowrward()
        self.pe_layer = PositionalEncoding().forward()
        self.multi_head_attention_layer = MultiHeadAttnetion().forward()
        self.add_norm_layer = AddAndNorm().forward()
        self.ff_layer = PositionWiseFeedFoward().forward()
        self.add_norm_layer2 = AddAndNorm().forward()

        # 임베딩
        # 포지셔널 임베딩
        # multi head 어텐션
        # add norm
        # ff
        # add norm
    
    def forward(self, x):
        # 임베딩
        # 
        pass


class DecoderLayer(nn.Module):
    def __init__(self, ):
        super(DecoderLayer, self).__init__()
        # 임베딩
        # 포지셔널 임베딩
        # masked multi head 어텐션
        # add norm
        # multi head 어텐션(Q, K는 encoderlayer output)
        # add norm
        # ff
        # add norm
    
    def forward(self, ):
        pass


class MultiHeadAttnetion(nn.Module):
    def __init__(self, d_model, d_k, d_v, h):
        super(MultiHeadAttnetion, self).__init__()
        self.h = h
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_model, h * d_k, bias=False)
        self.w_k = nn.Linear(d_model, h * d_k, bias=False)
        self.w_v = nn.Linear(d_model, h * d_v, bias=False)
        self.w_o = nn.Linear(h * d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(d_k)

    def _split_into_heads(self, *xs):
        # x : [BATCH * SEQ_LEN * D_MODEL] -> [BATCH * H * SEQ_LEN * D]
        return [x.view(x.size(0), x.size(1), self.h, -1).transpose(1,2) for x in xs]

    def forward(self, q, k, v, mask=None):
        # q, k, v : [BATCH * SEQ_LEN * D_MODEL]
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self._split_into_heads(q, k, v)

        x = self.attention(q,k,v,mask)
        x = x.transpose(1, 2).reshape(x.size(0), x.size(2), -1)  # -> x : [BATCH * SEQ_LEN * D_MODEL]
        x = self.w_o(x)
        return x

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = d_k ** -0.5
    
    def forward(self, q, k, v, mask):
        # q, k, v : [BATCH * H * SEQ_LEN * D_K(D_V)]
        x = torch.matmul(q, k.transpose(-2,-1))
        x = x if mask is None else x.masked_fill(mask, float('-inf'))
        x = torch.matmul(torch.softmax(self.scale(x), dim=-1), v)
        return 


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super(PositionalEncoding, self).__init__()
        tmp_pe = np.array([[pos/(10000 ** (2 * i / d_model)) for i in range(d_model // 2)] for pos in range(max_seq_len)])
        pe = np.empty((max_seq_len, d_model))
        pe[:, 0::2], pe[:, 1::2] = np.sin(tmp_pe), np.cos(tmp_pe)
        self.register_buffer('positional_encoding', torch.FloatTensor(pe).unsqueeze(0))

    def forward(self, x):
        return self.pe[:, x.size(1)]
    

class Embedding(nn.Moudule):
    def __init__(self, input_layer, output_layer):
        super(Embedding, self).__init__()
        pass

    def fowrward(self, x, ):
        pass


class PositionWiseFeedFoward(nn.Module):
    def __init__(self, d_model, d_ff):
        # RELU(xW1 + b1)W2 + b2
        super(PositionWiseFeedFoward, self).__init__()
        self.linear_layer1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear_layer2 = nn.Linear(d_ff, d_model)


    def forward(self, x):
        # x -> mha output(seq_len, d_model)
        linear1 = self.linear_layer1(x)
        relu = self.relu(linear1)
        linear2 = self.linear_layer2(relu)
        return linear2


class AddAndNorm(nn.Module):
    def __init__(self, d_model, p_drop):
        super(AddAndNorm, self).__init__()
        self.norm_layer = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(p_drop)

    def forward(self, inputs, x):
        norm = self.norm_layer(inputs + self.dropout_layer(x))
        return norm