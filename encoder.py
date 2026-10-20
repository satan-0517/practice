#编码器层和完整编码器
import math
import torch
import torch.nn as nn

#从其它模块中导入需要的模块
from positional_encoding import PositionalEncoding
from attention import MultiHeadAttention
from feed_forward import PositionWiseFeedforward

class EncoderLayer(nn.Module):
    """
    单层编码器：自注意力+FFN,每个子层都有残差+LayerNorm（Pre-LN)
    """
    def __init__(self,
                 d_model:int=512,
                 n_heads:int =8,
                 d_ff:int =2048,
                 dropout:float=0.1):
        super().__init__()
        #多头注意力机制
        self.self_attn=MultiHeadAttention(d_model,n_heads,dropout)
        #前馈网络
        self.ffn=PositionWiseFeedforward(d_model,d_ff,dropout)
        #两个LayerNorm,分别用在自注意力和FFN之前（Pre-LN)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        #dropout层（这里只定义但没有直接使用）
        self.dropout=nn.Dropout(dropout)

    def forward(self,x:torch.Tensor,mask:torch.Tensor=None)->torch.Tensor:
        #子层1：多头注意力机制
        #保存残差
        residual=x
        #Pre-LN：先做LayerNorm，再进注意力
        x=self.norm1(x)
        #自注意力：q=k=v=x
        #返回值里我们只关心输出，注意力权重用_忽略
        x,_=self.self_attn(x,x,x,mask)
        #残差连接：原始输入+注意力输出
        x=residual+x

        #子层2：前馈网络
        #保存残差
        residual=x
        #Pre-LN:先做LayerNorm，再进FFN
        x=self.norm2(x)
        #FFN
        x=self.ffn(x)
        #残差连接
        x=x+residual
        #输出
        return x

class Encoder(nn.Module):
    """
    完整编码器：token embedding+编码器+N层
    """
    def __init__(self,
                 src_vocab_size:int, #源语言词表大小
                 d_model:int=512,
                 n_layers:int =6,   #堆叠多少层
                 n_heads:int=8,
                 d_ff:int =2048,
                 max_seq_len:int=5000,
                 dropout:float=0.1):
        super().__init__()
        #保存d_model,前向传播时用
        self.d_model=d_model
        #token嵌入层：把token_id映射成d_model维向量
        self.token_embedding=nn.Embedding(src_vocab_size,d_model)
        #位置编码层
        self.pos_encoding=PositionalEncoding(d_model,max_seq_len,dropout)
        #N个编码层，用ModuleList管理
        #nn.ModuleList是pytorch提供的转么用来存放nn.Module子模块的列表，他本身就是一个nn.Module
        self.layers=nn.ModuleList([EncoderLayer(d_model,n_heads,d_ff,dropout) for _ in range(n_layers)])
        #最后一层LayerNorm
        self.norm=nn.LayerNorm(d_model)

    def forward(self,x:torch.Tensor,mask:torch.Tensor=None)->torch.Tensor:
        #x:[B,src_seq_len]每个元素是token id
        #1.token嵌入，并乘以sqrt（d_model)缩放
        #缩放是为了让嵌入值和对应位置编码的量级匹配
        x=self.token_embedding(x)*math.sqrt(self.d_model)
        #2.加入位置编码
        x=x+self.pos_encoding(x)
        #3.依次通过N个编码器层
        for layer in self.layers:
            x=layer(x,mask)
        #4.最终的LayerNorm
        x=self.norm(x)
        #返回编码器输出：[B,src_seq_len,d_model]
        return x