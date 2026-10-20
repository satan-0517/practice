#模型超参数

src_vocab_size=37000    #源语言词表大小
tgt_vocab_size=37000    #目标语言词表大小
d_model=512     #模型隐藏维度，每个token用512维向量表示
n_layers=6      #编码器和解码器各堆叠6层
n_heads=8       #多头注意力有8头
d_ff=1048       #前馈网络中间层维度，512->2048->512
max_seq_len=512     #支持的最大序列长度，位置编码按这个长度预计算
dropout=0.1     #dropout概论，训练时随即丢弃10%的神经元
pad_token_id=0      #padding token的id，掩码时用它来判断那些是填充

#测试时用的参数
batch_size=2    #测试时一个batch有两条样本
src_seq_len=10  #测试时源序列长度
tgt_seq_len=8   #测试时目标序列长度