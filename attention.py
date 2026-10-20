#这个文件存放注意力相关的模块：缩放点积注意力+多头注意力

import math     #用来算sqrt(d_k)
import torch    #pytorch核心库
import torch.nn as nn   #神经网络模块，比如Linear，Dropout
import torch.nn.functional as F     #函数API，比如：softmax

"""
nn.Module是PyTorch中所有神经网络模块的基类,继承nn.Module之后。会自动获得很多能力
1.自动管理参数：在__init__里的nn.Linear、nn.embedding等层，会被自动注册为模型的参数
2.自动管理子模块：nn.Module可以嵌套
3.可以用model(x)调用：只要你定义了forward，python会自动调用model.forward()
4.可以切换训练/评估模式：midel.train():训练模式，Dropout生效，model.eval():评估模式，Dropout失效
5.可以保存和加载
6.可以把模型移到GPU
"""
class ScaledDotProductAttention(nn.Module):
    """
    公式：Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V
    """
    def __init__(self,dropout:float=0.1):   #初始化方法，创建对象时自动调用
        #调用父类nn.Module的构造函数，必须写
        super().__init__()
        #定义一个Dropout层，作用在注意力权重上，防止过拟合
        self.dropout=nn.Dropout(dropout)

    # forward：自己定义的方法，用来描述前向传播的计算过程
    def forward(self,
                q:torch.Tensor,       #查询张量[B,n_heads,seq_len_q,d_q],B:batch-size:一批多少样本
                k:torch.Tensor,       #查询张量 [B,n_heads,seq_len_k,d_k]，n_heads:多少个注意力头
                v:torch.Tensor,       #查询张量[B,n_heads,seq_len_v,d_v],seq_len_q:query的序列长度,d_k:每个头的query/key维度
                mask:torch.Tensor=None):        #默认为None，表示不传参数，不屏蔽任何位置
        #取q的最后一个维度，也就是d_q
        d_k=q.size(-1)

        #1.Q和K做矩阵乘法
        #k.transpose(-2,-1)把最后两个维度交换：[B,n_heads,seq_len_q,d_q]->[B,n_heads,d_q,seq_len_q]
        #相乘后得到[B,n_heads,seq_len_q,seq_len_q],即每个query对每个key的相似度
        #再除以sqrt(d_k)做缩放，避免点积结果太大导致softmax梯度消失
        # torch.matmul 的批量矩阵乘法规则：它把前面的维度当作"批量维度"，只对最后两维做矩阵乘法
        scores=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(d_k)

        #2.如果有mask，把需要屏蔽的位置设成-1e9
        #mask_fill会把mask设为Ture的位置替换成-1e9
        #softmax之后-1e9对应的权重几乎为0，相当于忽略这些位置
        if mask is not  None:
            scores=scores.masked_fill(mask,-1e9)

        #3.在最后一个维度(key维度)做softmax，得到注意力权重
        #每一行加起来等于1，表示query对所有key的注意力分布
        attn_weights=F.softmax(scores,dim=-1)

        #4.对注意力权重做dropout，训练时随机置零一部分
        attn_weights=self.dropout(attn_weights)

        #5.注意力权重和v相乘，得到加权求和的结果
        #形状：[B,n_heads,seq_len_q,d_q]@[B,n_heads,d_k,seq_len_k]->[B,n_heads,seq_len_q,d_v]
        # torch.matmul 的批量矩阵乘法规则：它把前面的维度当作"批量维度"，只对最后两维做矩阵乘法
        output=torch.matmul(attn_weights,v)

        #返回输出和注意力权重（权重可视化）
        return output,attn_weights

class MultiHeadAttention(nn.Module):
    #MultiHead(Q,K,V)=Concat(head_1,head_2,...head_h)W^O
    def __init__(self,
                d_model:int=512,
                n_heads:int=8,
                dropout:float=0.1):
        super().__init__()
        #断言：d_model必须能被n_heads整除，否则没法平均分给每个头
        assert d_model%n_heads==0,"d_model必须能被N_heads整除"

        self.d_model=d_model    #保存总维数
        self.n_heads=n_heads    #保存头数
        self.d_k=d_model//n_heads   #每个头的Q/K的维度
        self.d_v=d_model//n_heads   #每个头的V维度

        #四个线性层，把输入 投影到Q，K，V，空间，最后在投影回d_model
        self.w_q=nn.Linear(d_model,d_model)#W_Q,生成Q，生成所有头的Q
        self.w_k=nn.Linear(d_model,d_model)#W_K,生成K，生成所有头的K
        self.w_v=nn.Linear(d_model,d_model)#W_V,生成V，生成所有头的V
        self.w_o=nn.Linear(d_model,d_model)#输出投影

        #缩放点积注意力模块
        self.attention=ScaledDotProductAttention(dropout)
        #输出端的dropout
        self.dropout=nn.Dropout(dropout)

    def forward(self,
                q:torch.Tensor,     #[B,seq_len_q,d_model]，输入
                k:torch.Tensor,     #[B,seq_len_k,d_model]，输入
                v:torch.Tensor,     #[B,seq_len_v,d_model]，输入
                mask:torch.Tensor=None):    #[B,1,seq_len_q,seq_len_v]
        batch_size=q.size(0)

        #1.线性投影，把Q，K，V，都投影到d_model维度
        #注意：q，k,v,可能来子不同的地方（自注意力时相同，交叉注意力时不同）
        q=self.w_q(q)   #self.w_q是一个Linear层，self.w_q(q)十八输入q传入线性层做前向计算
        k=self.w_k(k)   #得到所有头的K
        v=self.w_v(v)   #得到所有头的V

        #拆分成多头
        #view:把d_model拆分成n_heads x d_k
        #[B,seq_len,d_model]->[B,seq_len,n_heads,d_k]
        #transpose(1,2):把n_heads换到前面
        q=q.view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)
        k=k.view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)
        v=v.view(batch_size,-1,self.n_heads,self.d_v).transpose(1,2)

        #3.对每个头做缩放点积注意力
        #attn_output:[B,n_heads,seq_len_q,d_v]
        #attn_weights:[B,n_heads,seq_len_q,seq_len_k]
        attn_output,attn_weights=self.attention(q,k,v,mask)

        #4.把多头结果拼回去
        #transpose(1,2):[B,n_heads,seq_len,,d_v]->[B,seq_len,n_heads,d_v]
        #contiguous():让内存连续，transpose后内存不连续，view需要连续内存
        #view:把n_heads x d_v合并成d_model
        #   [B, seq_len, n_heads, d_v] -> [B, seq_len, d_model]
        attn_output=attn_output.transpose(1,2).contiguous().view(batch_size,-1,self.d_model)

        #5.最终线性投影W_O
        output=self.w_o(attn_output)
        #输出output
        output=self.dropout(output)

        #返回输出和注意力权重
        return output,attn_weights


