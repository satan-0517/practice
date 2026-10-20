#把编码器，解码器，输出层组合成完整的transformer

import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size:int ,
                 tgt_vocab_size:int,
                 d_model:int =512,
                 n_layers:int =6,
                 n_heads:int=8,
                 d_ff:int =2048,
                 max_seq_len:int=5000,
                 dropout:float=0.1,
                 pad_token_id:int =0):
        super().__init__()
        #保存pad_token_id，创建掩码时要用
        self.pad_token_id=pad_token_id

        #编码器
        self.encoder=Encoder(src_vocab_size,d_model,n_layers,n_heads,d_ff,max_seq_len,dropout)
        #解码器
        self.decoder=Decoder(tgt_vocab_size,d_model,n_layers,n_heads,d_ff,max_seq_len,dropout)

        #输出线性层：把d_model维映射到目标词表大小
        #输出每个位置在此表上的logits
        self.fc=nn.Linear(d_model,tgt_vocab_size)

        #论文中的权重共享：src_vocab_size=tgt_vocab_size
        #让嵌入层和输出层共享同一个权重矩阵，减少参数
        if src_vocab_size==tgt_vocab_size:
            self.encoder.token_embedding.weight=self.fc.weight
            self.decoder.token_embedding.weight=self.fc.weight

        #参数初始化
        self._init_parameters()

    def _init_parameters(self):
        #遍历所有参数
        for p in self.parameters():
            #只对维度大于1的参数，即矩阵做Xavier初始化
            #偏置和LayerNorm的参数不重要
            if p.dim()>1:
                nn.init.xavier_uniform_(p)


    def create_padding_mask(self,seq:torch.Tensor)->torch.Tensor:
        """
               创建 padding 掩码：padding 位置为 True
               seq: [B, seq_len]
               返回: [B, 1, 1, seq_len]
        """
        #seq==pad_token_id,得到[B，seq_len]
        #unsqueeze(1)->[B,q,seq_len]
        #unsuqeeze(2)->[B,1,1,seq_len]
        #这样可以广播到[B,n_heads,seq_len_q,seq_len_k]
        mask=(seq==self.pad_token_id).unsqueeze(1).unsqueeze(2)
        return mask

    def create_look_ahead_mask(self,seq_len:int)->torch.Tensor:
        """
                创建前瞻掩码：上三角为 True，防止看到未来 token
                返回: [1, 1, seq_len, seq_len]
        """
        #torch.ones(seq_len,seq_len)全1矩阵
        #triu(diagonal=1)取上三角不包含对角线
        #   [[0, 1, 1],
        #    [0, 0, 1],
        #    [0, 0, 0]]
        # .bool() 转成布尔
        mask=torch.triu(torch.ones(seq_len,seq_len),diagonal=1).bool()
        # 加两个维度，变成 [1, 1, seq_len, seq_len]
        return mask.unsqueeze(0).unsqueeze(0)


    def forward(self,src_seq:torch.Tensor,tgt_seq:torch.Tensor)->torch.Tensor:
        """
                src_seq: [B, src_seq_len]  源语言 token id
                tgt_seq: [B, tgt_seq_len]  目标语言 token id
                返回:     [B, tgt_seq_len, tgt_vocab_size]
        """
        #1.创建编码器的padding掩码
        enc_padding_mask=self.create_padding_mask(src_seq)
        #2.编码器前向传播
        enc_output=self.encoder(src_seq,enc_padding_mask)
        #3.创建解码器的掩码
        tgt_seq_len=tgt_seq.size(1)
        #3.1前瞻掩码
        look_ahead_mask=self.create_look_ahead_mask(tgt_seq_len).to(tgt_seq.device)
        #3.2目标序列的padding掩码
        tgt_padding_mask=self.create_padding_mask(tgt_seq)
        #3.3合并两个掩码，只要有一个为true就mask
        #torch.max对布尔张量相当于逻辑或
        dec_self_attn_mask=torch.max(look_ahead_mask,tgt_padding_mask)

        #4.交叉注意力掩码：屏蔽编码器的padding token
        dec_cross_attn_mask=self.create_padding_mask(src_seq)

        #5.解码器的前向传播
        dec_output=self.decoder(tgt_seq,enc_output,dec_self_attn_mask,dec_cross_attn_mask)
        #6.输出线性层，得到每个位置在词表上的logits
        output=self.fc(dec_output)
        # 返回 logits，形状 [B, tgt_seq_len, tgt_vocab_size]
        return output





