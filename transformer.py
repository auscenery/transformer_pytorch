#!/usr/bin/env pytho

# @email: liangchuanjian@wps.cn
# @date: 2019-07-07 15:19:00
# @reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html

import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from common import LayerNorm, clones

## ------------------------------------------encoder-decoder(seq2seq) architecture implemention------------------------------------------
class EncoderDecoder(nn.Module):
    """
        A standard encoder-decoder architecture
    """
    def __init__(self, encoder, decoder, src_embedding, target_embedding, output_layer):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.target_embedding = target_embedding
        self.output_layer = output_layer

    def forward(self, src, target, src_mask, target_mask):
        context_memory = self.encode(src, src_mask)
        return self.decode(context_memory, src_mask, target, target_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embedding(src), src_mask)

    def decode(self, context_memory, src_mask, target, target_mask):
        return self.decoder(self.target_embedding(target), context_memory, src_mask, target_mask)


class OutputLayer(nn.Module):
    """
        define standard linear + softmax output_layer
    """
    def __init__(self, d_model, vocab_size):
        super(OutputLayer, self).__init__()
        self.proj_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.proj_layer(x), dim=-1)


## -------------------------------------------------sublayer-connection------------------------------------------------
class SublayerConnection(nn.Module):
    """
        a residual connection followed by layer norm.
        note for code simplicity the norm is first as opposed to last
    """
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer: nn.Module):
        # layer normalization && residual connection add operation
        sublayer_x = sublayer(self.norm(x))
        return x+ self.dropout(sublayer_x)


##---------------------------------------------------encoder----------------------------------------------------------
class Encoder(nn.Module):
    """
        Core encoder is a stack of N layers
    """
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)
    
    def forward(self, x, mask):
        """
            pass the input (and mask) through each layer in turn
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x) # 最后一层输出的时候再做一次layer normalization

class EncoderLayer(nn.Module):
    """
        Each encoder layer of stacks has two sub-layers. 
        The first is a multi-head self-attention mechanism, 
        and the second is a simple, position-wise fully connected feed- forward network,
        之所以是position-wise是因为处理的attention输出是某一个位置i的attention输出。
    """
    def __init__(self, d_model, self_attention_layer, feed_forward_layer, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attention_layer = self_attention_layer
        self.feed_forward_layer = feed_forward_layer
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model
    
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attention_layer(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward_layer)


## ----------------------------------------------decoder-----------------------------------------------
class Decoder(nn.Module):
    """
        core N layer decoder with masking
    """
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, context_memory, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, context_memory, src_mask, target_mask)
        return self.norm(x) # 最后一层输出的时候再做一次layer normalization


class DecoderLayer(nn.Module):
    """
        In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, 
        which performs multi-head attention over the output of the encoder stack. 
        Similar to the encoder, we employ residual connections around each of the sub-layers, 
        followed by layer normalization.
    """
    def __init__(self, d_model, self_attention_layer, src_attention_layer, feed_forward_layer, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention_layer = self_attention_layer
        self.src_attention_layer = src_attention_layer
        self.feed_forward_layer = feed_forward_layer
        self.sublayer = clones(SublayerConnection(d_model, dropout), 3)
        self.d_model = d_model

    def forward(self, x, context_memory, src_mask, target_mask):
        x = self.sublayer[0](x, lambda x: self.self_attention_layer(x, x, x, target_mask))  # query, key, value
        x = self.sublayer[1](x, lambda x: self.src_attention_layer(x, context_memory, context_memory, src_mask))
        return self.sublayer[2](x, self.feed_forward_layer)


##-------------------------------------------------attention---------------------------------------------------------------------
def compute_attention(query, key, value, mask=None, dropout_layer=None):
    """
        @param: query, key, value, mask, types all are torch.Tensor
        Compute 'Scaled Dot Product Attention': Attention(Q, K, V) = softmax(Q * K/sqrt(dk)) * V
        While for small values of dk the two mechanisms perform similarly, additive attention outperforms dot product attention 
        without scaling for larger values of dk (cite). We suspect that for large values of dk, the dot products grow large 
        in magnitude, pushing the softmax function into regions where it has extremely small gradients 
        (To illustrate why the dot products get large, assume that the components of q and k are independent random variables with
         mean 0 and variance 1. Then their dot product, q⋅k=∑qiki, has mean 0 and variance dk.). 
         To counteract this effect, we scale the dot products by 1/sqrt(dk).
    """
    dk = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dk)
    if mask is not None:
        #　由于模型是以batch为单位进行训练的，batch的句长以其中最长的那个句子为准，其他句子要做padding。padding项在计算的过程中如果不处理的话，
        # 会引入噪音，所以就需要mask，来使padding项不对计算起作用。mask在attention机制中的实现非常简单，就是在softmax之前，
        # 把padding位置元素加一个极大的负数，强制其softmax后的概率结果为0。
        # 举个例子，[1,1,1]经过softmax计算后结果约为[0.33,0.33,0.33]，[1,1,-1e9] softmax的计算结果约为[0.5, 0.5,0]。这样就相当于mask掉了数组中的第三项元素。
        scores = scores.masked_fill(mask == 0, -1e9)    # 屏蔽机制，近似表示为负无穷
    p_atten = F.softmax(scores, dim=-1)
    if dropout_layer is not None:
        p_atten = dropout_layer(p_atten)
    return torch.matmul(p_atten, value), p_atten


class MultiHeadedAttention(nn.Module):
    """
        Multi-head attention allows the model to jointly attend to information from different representation subspaces 
        at different positions. 
    """
    def __init__(self, header_num, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % header_num == 0
        self.dk = d_model // header_num
        self.header_num = header_num
        self.linear_layers = clones(nn.Linear(d_model, d_model), 4)
        self.atten = None
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # 没看明白
        if mask is not None:
            mask = mask.unsqueeze(dim=1) # expand a dimension at dim 1(axis=1)
        nbatches = query.size(0)

        # 多个头通过矢量化计算，变成并行计算，加快速度
        # view function is equal to reshape tensor
        ## 稍微有些没看懂
        query, key, value = [
            linear_layer(x).view(nbatches, -1, self.header_num, self.dk).transpose(1, 2)
            for linear_layer, x in zip(self.linear_layers, (query, key, value))
        ]

        x, self.atten = compute_attention(query, key, value, mask, self.dropout_layer)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.header_num * self.dk)

        return self.linear_layers[-1](x)


## ---------------------------------------------FFN-----------------------------------------------------------------
class PositionWiseFeedForward(nn.Module):
    """
        implements FFN network, FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.ff1_layer = nn.Linear(d_model, d_ff)
        self.ff2_layer = nn.Linear(d_ff, d_model)
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x):   # x.shape: [batch_size, seq_len, d_model]
        relu_x = F.relu(self.ff1_layer(x))
        dropout_x = self.dropout_layer(relu_x)
        return self.ff2_layer(dropout_x)

## ----------------------------------------------Ebemdding and positional encoding--------------------------------------
class Embeddings(nn.Module):
    def __init__(self, d_model, d_vocab):
        super(Embeddings, self).__init__()
        self.embed_layer = nn.Embedding(d_vocab, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        return self.embed_layer(x) * math.sqrt(self.d_model)    # 对每一维度进行缩放sqrt(d_model, 防止每一维度太大，落入非线性函数的饱和区域)


class PositionalEncoding(nn.Module):
    """
        implement the PE function: 
        PE(pos, 2i)     = sin(pos/pow(10000, 2*i/d_model)),
        PE(pos, 2i + 1) = cos(pos/pow(10000, (2*i)/d_model)),
        note: 序列上不同位置上的x对应的同一个维度i(0<= i <= d_model)，也就是同一个维度特征i上，才具有线性关系
        sin(a+b) = sin(a)cos(b) + cos(a)sin(b)
        PE(p+k, 2i) = PE(p, 2i)PE(k, 2i+1) + PE(p, 2i+1)PE(k, 2i)
        另外一个原因是cos,sin的值域在[-1,1]之间，也方便赋值给权重参数
    """
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout_layer = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)   # 不能写0， 否则报类型错误
        div_term = torch.exp(torch.arange(0., d_model, 2) * -math.log(10000.0) / d_model)
        # div_term = 1 / (10000 ** (torch.arange(0., d_model, 2) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # 增加一个维度batch_size, 利用向量相加广播机制
        self.register_buffer('pe', pe) # 不作为变量，不需要学习的，持久化放在内存中，通过名字来存取

    def forward(self, x):   # x.shape: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False) # 取出实际的长度
        return self.dropout_layer(x)


##-----------------------------------------------make transformer---------------------------------------------------------------
def make_transformer(src_vacab_size, target_vocab_size, N=6, d_model=512, d_ff=2048, header_num=8, dropout=0.1):
    # 首先: 生成各个基本的积木块单元
    atten_layer = MultiHeadedAttention(header_num, d_model, dropout)
    ffn_layer = PositionWiseFeedForward(d_model, d_ff, dropout)
    position_layer = PositionalEncoding(d_model, dropout)
    
    # 然后:分别定义encoder和decoder
    encoder_layer = EncoderLayer(d_model, copy.deepcopy(atten_layer), copy.deepcopy(ffn_layer), dropout)
    encoder = Encoder(encoder_layer, N)
    decoder_layer = DecoderLayer(d_model, copy.deepcopy(atten_layer), copy.deepcopy(atten_layer), copy.deepcopy(ffn_layer), dropout)
    decoder = Decoder(decoder_layer, N)

    # 接着：定义encoder embedding 和　decoder embedding, 以及decoder output layer
    src_embedding = nn.Sequential(Embeddings(d_model, src_vacab_size), copy.deepcopy(position_layer))
    target_embedding = nn.Sequential(Embeddings(d_model, target_vocab_size), copy.deepcopy(position_layer))
    output_layer = OutputLayer(d_model, target_vocab_size)
    
    # 最后：实例化transformer　architecture model和初始化模型的所有参数
    model = EncoderDecoder(encoder, decoder, src_embedding, target_embedding, output_layer)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return model