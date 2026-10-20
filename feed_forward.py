#逐位置前馈网络，就是两个嵌入层+ReLU
import torch
import torch.nn as nn

class PositionWiseFeedforward(nn.Module):
    """
    逐位置前馈网络
    公式：FFN(x)=max(0,xW1+b1)W2+b2
    """
    def __init__(self,
                 d_model:int =512,
                 d_ff:int =2048,
                 dropout:float=0.1):
        super().__init__()
        #第一个线性层，升维，512->2048
        self.fc1=nn.Linear(512,2048)
        #第二个线性层，降维，2048->512
        self.fc2=nn.Linear(2048,512)
        #dropout层
        self.dropout=nn.Dropout(dropout)
        #ReLU激活函数，max(0,x)
        self.activation=nn.ReLU()
    #->torrch.Tensor是python的类型注解，用来标注函数的输入输出类型，不影响运行，只是给人看的说明
    def forward(self,x:torch.Tensor)->torch.Tensor:
        """
        x:[B,seq_len,d_model]
        """
        #升维
        x=self.fc1(x)
        #激活函数
        x=self.activation(x)
        #dropout：在训练时随机把一部分的神经元的输出设置维0，概率维p，防止过拟合
        x=self.dropout(x)
        #降维
        x=self.fc2(x)
        #返回结果
        return x



