#测试模型是否能前向传播
import torch
import config
from model import Transformer

def main():
    #1.初始化模型
    model=Transformer(
        src_vocab_size=config.src_vocab_size,
        tgt_vocab_size=config.tgt_vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
        pad_token_id=config.pad_token_id
    )
    #2.随机生成测试数据
    #torch.randint(low,high,size):它是 PyTorch 里用来生成随机整数张量的函数,范围是 [low, high)，左闭右开,size 是输出张量的形状
    src_seq=torch.randint(
        1,config.src_vocab_size,(config.batch_size,config.tgt_seq_len)
    )
    tgt_seq=torch.randint(
        1,config.tgt_vocab_size,(config.batch_size,config.tgt_seq_len)
    )
    #3.切换到评估模式，关闭dropout
    model.eval()
    #4.关闭梯度计算，节省内存，加速
    with torch.no_grad():
        #5.前向传播
        output=model(src_seq,tgt_seq)

        # 6. 打印形状
    print(f"源序列形状: {src_seq.shape}")
    print(f"目标序列形状: {tgt_seq.shape}")
    print(f"模型输出形状: {output.shape}")
    print("✅ 模型前向传播测试通过！")

# 只有直接运行这个文件时才执行 main()
# 被 import 时不会执行
if __name__ == "__main__":
    main()