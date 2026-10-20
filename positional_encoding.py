#正弦余弦位置编码，给模型注入位置信息

import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    PE(pos,2i)=sin(pos/10000^(2i/d_model))
    PE(pos,2i+1)=cos(pos/10000^(2i/d_model))
    """
    def __init__(self,
                d_model:int=512,    #编码维度
                max_seq_len:int=5000,   #支持的最大序列长度
                dropout:float=0.1):
        super().__init__()
        #dropout层
        self.dropout=nn.Dropout(dropout)

        #创建一个全零矩阵，用来存放位置编码，形状[max_seq_len,d_model]
        pe=torch.zeros(max_seq_len,d_model)

        #位置索引0，1，2，...max_seq_len-1
        #arange生成一张一维张量，unsqueeze(1)变为列向量[max_seq_len,1]
        #torch.arange是PyTorch里生成等差数列的函数，返回的是张量
        #第一个参数0：起始值，第二个参数：结束值（不包括）第三个参数：步长（省略），dtype指定类型
        #unsqueeze(dim)的作用是在指定维度上插入一个大小为1的新维度
        position=torch.arange(0,max_seq_len,dtype=torch.float).unsqueeze(1)

        #计算分母项10000^(2i/d_model)
        #用exp和log是为了数值稳定
        #10000^(2i/d_model)=exp((2i/d_model)*log(10000))
        #这里取负号，因为原公式是除以10000^(2i/d_model)
        #arange(0,d_model,2)生成0，2，4，...,d_model-2，共d_model/2个
        #结果div_term形状[d_model/2]
        div_term=torch.exp(
            torch.arange(0,d_model,2).float()*(-math.log(10000)/d_model)
        )
        #偶数维度用sin:pe[:,0],pe[:,2],pe[:,4]....
        #position*div_term利用广播，形状[max_seq_len,d_model/2]
        pe[:,0::2]=torch.sin(position*div_term)
        #奇数维度用cos:pe[:,1],pe[:,3],pe[:5],...
        pe[:,1::2]=torch.cos(position*div_term)

        #在最前面加入一个batch维度，方便输入相加
        #[max_seq_len,d_model]->[B,max_seq_len,d_model]
        pe=pe.unsqueeze(0)

        #注册为buffer：不参与梯度更新，但会随模型一起保存、加载
        self.register_buffer("pe",pe)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        """
        x:[B,max_seq_len,d_model]
        """
        #取输入的序列长度
        seq_len=x.size(1)
        #取对应长度的位置编码加到输入上
        #self.pe[:,:seq_len,:]:形状[1,seq_len,d_model]广播到[B,seq_len,d_model]
        #.detach()确保位置编码不会被误当成需要梯度的参数
        x=x+self.pe[:,:seq_len,:].detach()
        #应用dropout
        x=self.dropout(x)
        return x



