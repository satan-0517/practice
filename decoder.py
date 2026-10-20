#解码层+完整解码器

import math
import torch
import torch.nn as nn

from attention import MultiHeadAttention
from feed_forward import PositionWiseFeedforward
from positional_encoding import  PositionalEncoding

class DecoderLayer(nn.Module):
    """
    单个解码器层：掩码自注意力+交叉注意力+FFN
    """
    def __init__(self,
                 d_model:int =512,
                 n_heads:int=8,
                 d_ff:int=2048,
                 dropout:float=0.1):
        super().__init__()
        #1.带掩码的子注意力（防止看到未来token）
        self.self_attn=MultiHeadAttention(d_model,n_heads,dropout)
        #2.交叉注意力：Q来自解码器，K|V来自编码器
        self.cross_attn=MultiHeadAttention(d_model,n_heads,dropout)
        #3.前馈网络
        self.ffn=PositionWiseFeedforward(d_model,d_ff,dropout)
        #4.三个LayerNorm
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.norm3=nn.LayerNorm(d_model)
        #dropout层
        self.dropout=nn.Dropout(dropout)

    def forward(self,
                x:torch.Tensor,     #解码器输入[B,tgt_seq_len,d_model]
                enc_output:torch.Tensor,     #编码器输出[B,src_seq_len,d_model]
                look_ahead_mask:torch.Tensor=None,      #前瞻掩码
                padding_mask:torch.Tensor=None)->torch.Tensor:      #编码器padding掩码
        #1.掩码自注意力
        residual=x
        x=self.norm1(x)
        x,_=self.self_attn.forward(x,x,x,look_ahead_mask)
        x=x+residual
        #2.交叉注意力
        residual=x
        x=self.norm2(x)
        x,_=self.cross_attn(x,enc_output,enc_output,padding_mask)
        x=x+residual
        #3.FFN
        residual=x
        x=self.norm3(x)
        x=self.ffn(x)
        x=x+residual
        #dropout(不用)

        return x

class Decoder(nn.Module):
    """
    完整解码器：token embedding+位置编码+层
    """
    def __init__(self,
                 tgt_vocab_size:int,
                 d_model:int =512,
                 n_layers:int=6,
                 n_heads:int=8,
                 d_ff:int=2048,
                 max_seq_len:int=5000,
                 dropout:float=0.1):
        super().__init__()
        self.d_model=d_model
        #目标语言token嵌入
        self.token_embedding=nn.Embedding(tgt_vocab_size,d_model)
        #位置编码
        self.pos_encoding=PositionalEncoding(d_model,max_seq_len,dropout)
        #N个解码器
        self.layers=nn.ModuleList([DecoderLayer(d_model,n_heads,d_ff,dropout)for _ in range (n_layers)])
        #最终的LayerNorm
        self.norm=nn.LayerNorm(d_model)

    def forward(self,
                x:torch.Tensor,
                enc_output:torch.Tensor,
                look_ahead_mask:torch.Tensor=None,
                padding_mask:torch.Tensor=None)->torch.Tensor:
        #1.token嵌入+缩放
        x=self.token_embedding(x)/math.sqrt(self.d_model)
        #2.位置编码
        x=self.pos_encoding(x)
        #3.依次通过N个解码器层
        for layer in self.layers:
            x=layer(x,enc_output,look_ahead_mask,padding_mask)
        #4.最终LayerNorm
        x=self.norm(x)
        #返回解码器输出：[B,tgt_seq_len,d_model]
        return x





