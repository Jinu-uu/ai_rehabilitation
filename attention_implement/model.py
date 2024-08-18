import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransFormerModel(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_padding_idx, trg_padding_idx,
                 max_seq_len=5000, n=6, d_model=512, d_ff=2048, h=8, d_k=64, d_v=64, p_drop=0.1):
        super(TransFormerModel,self).__init__()
        # 인코더 디코더 임베딩 및 포지셔널 인코딩
        # 인코더
        # 디코더
        # 디코더 output linear
        # softmax
        self.src_embedding_layer = Embedding(src_vocab_size, d_model, src_padding_idx)
        self.trg_embedding_layer = Embedding(trg_vocab_size, d_model, trg_padding_idx)
        self.encoder = EncoderStack(n, d_model, d_k, d_v, h, d_ff, p_drop)
        self.decoder = DecoderStack(n, d_model, d_k, d_v, h, d_ff, p_drop)
        self.dropout = nn.Dropout(p_drop)
        self.embedding_layer = Embedding(src_vocab_size, d_model, src_padding_idx)
        self.pe_layer = PositionalEncoding(max_seq_len, d_model)
        self.projection = nn.Linear(d_model, trg_vocab_size, bias=False)

        self.register_buffer('leftward_mask', torch.triu(torch.ones((max_seq_len, max_seq_len)), diagonal=1).bool())

    @classmethod
    def _mask_paddings(cls, x, padding_idx):
        return x.eq(padding_idx).unsqueeze(-2).unsqueeze(1)

    def forward(self, src, trg):
        # src : [BATCH * SRC_SEQ_LEN], trg : [BATCH * TRG_SEQ_LEN]

        # Mask
        src_padding_mask = self._mask_paddings(src, self.src_padding_idx)
        trg_padding_mask = self._mask_paddings(trg, self.trg_padding_idx)
        trg_self_attn_mask = trg_padding_mask | self.leftward_mask[:trg.size(-1), :trg.size(-1)]

        src = self.src_embedding_layer(src)
        src = self.dropout(src + self.pe_layer(src))

        x = self.encoder(src, padding_mask = src_padding_mask)

        trg = self.trg_embedding_layer(trg)
        trg = self.dropout(trg + self.pe_layer(trg))

        x = self.decoder(trg, x, padding_mask = trg_self_attn_mask, enc_dec_attn_mask = src_padding_mask)

        x = self.projection(x)
        return x



class EncoderStack(nn.Module):
    def __init__(self, n, d_model, d_k, d_v, h, d_ff, p_drop):
        super(EncoderStack, self).__init__()
        self.encoder_layer_list = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, p_drop) for _ in range(n)])

    def forward(self, x, padding_mask):
        for encoder_layer in self.encoder_layer_list:
            x = encoder_layer(x, padding_mask)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, d_ff, p_drop):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention_layer = MultiHeadAttnetion(d_model, d_k, d_v, h)
        self.add_norm_layer = AddAndNorm(d_model, p_drop)
        self.ff_layer = PositionWiseFeedFoward(d_model, d_ff)
        self.add_norm_layer2 = AddAndNorm(d_model, p_drop)

        # multi head 어텐션
        # add norm
        # ff
        # add norm
    
    def forward(self, x, padding_mask):
        x = self.add_norm_layer(x, self.multi_head_attention_layer(x, x, x, mask=padding_mask))
        x = self.add_norm_layer2(x, self.ff_layer(x))
        return x


class DecoderStack(nn.Module):
    def __init__(self, n, d_model, d_k, d_v, h, d_ff, p_drop):
        super(DecoderStack, self).__init__()
        self.decoder_layer_list = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, p_drop) for _ in range(n)])

    def forward(self, x, x_enc, self_attn_mask, enc_dec_attn_mask):
        for decoder_layer in self.decoder_layer_list:
            x = decoder_layer(x, x_enc, self_attn_mask, enc_dec_attn_mask)



class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, d_ff, p_drop):
        super(DecoderLayer, self).__init__()
        # masked multi head 어텐션
        # add norm
        # multi head 어텐션(Q, K는 encoderlayer output)
        # add norm
        # ff
        # add norm
        self.multi_head_attention_layer = MultiHeadAttnetion(d_model, d_k, d_v, h)
        self.add_norm_layer = AddAndNorm(d_model, p_drop)
        self.multi_head_attention_layer2 = MultiHeadAttnetion(d_model, d_k, d_v, h)
        self.add_norm_layer2 = AddAndNorm(d_model, p_drop)
        self.ff_layer = PositionWiseFeedFoward(d_model, d_ff)
        self.add_norm_layer3 = AddAndNorm(d_model, p_drop)
    
    def forward(self, x, x_enc, self_attn_mask, enc_dec_attn_mask):
        x = self.add_norm_layer(x, self.multi_head_attention_layer(x, x, x, mask=self_attn_mask))
        x = self.add_norm_layer2(x, self.multi_head_attention_layer2(x,x_enc, x_enc, mask=enc_dec_attn_mask))
        x = self.add_norm_layer3(x, self.ff_layer(x))
        return x

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
    

class Embedding(nn.Module):
    def __init__(self, src_vocab_size, d_model, src_padding_idx):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(src_vocab_size, d_model, src_padding_idx)
        self.scale = d_model ** 0.5
        pass

    def fowrward(self, x):
        x = self.embedding(x)
        return x * self.scale



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