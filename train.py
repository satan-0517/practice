# train.py
# 训练脚本

import math                     # sqrt
import time                     # 计时
import torch                    # PyTorch 核心
import torch.nn as nn           # 神经网络模块

import config                   # 超参数
from model import Transformer   # 主模型
from dataset import SyntheticDataset, build_dataloader  # 数据集
from utils import NoamScheduler, get_device              # 工具


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion,
                    device, pad_token_id, clip_grad=1.0, log_every=50):
    """
    训练一个 epoch
    :param model: Transformer
    :param dataloader: DataLoader
    :param optimizer: 优化器
    :param scheduler: 学习率调度器
    :param criterion: 损失函数
    :param device: 设备
    :param pad_token_id: padding id
    :param clip_grad: 梯度裁剪阈值
    :param log_every: 每多少步打印一次
    :return: 平均 loss
    """
    # 切换到训练模式：启用 dropout
    model.train()
    # 累计 loss
    total_loss = 0.0
    # 累计步数
    total_steps = 0
    # 记录开始时间
    start_time = time.time()

    # 遍历数据
    for step, (src, tgt) in enumerate(dataloader):
        # 把数据放到设备上
        src = src.to(device)
        tgt = tgt.to(device)

        # 构造解码器输入和标签
        # 解码器输入 = <bos> + tgt[:-1]
        # 这里简化处理：直接用 tgt 作为输入，标签也是 tgt
        # 严格来说应该错一位，下面用 tgt_input 和 tgt_label 区分
        # tgt_input: 去掉最后一个 token
        # tgt_label: 去掉第一个 token
        tgt_input = tgt[:, :-1]   # [B, L-1]
        tgt_label = tgt[:, 1:]    # [B, L-1]

        # 前向传播
        logits = model(src, tgt_input)  # [B, L-1, vocab_size]

        # 计算损失
        # CrossEntropyLoss 要求：
        #   input:  [N, C] 或 [B, C, L]
        #   target: [N] 或 [B, L]
        # 所以把 logits 转成 [B*L, vocab] 的形式
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),  # [B*(L-1), vocab]
            tgt_label.reshape(-1)                 # [B*(L-1)]
        )

        # 清空梯度
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        # 更新参数
        optimizer.step()
        # 更新学习率
        scheduler.step()

        # 累计 loss
        total_loss += loss.item()
        # 累计步数
        total_steps += 1

        # 打印日志
        if (step + 1) % log_every == 0:
            avg_loss = total_loss / total_steps
            lr = scheduler.get_last_lr()
            elapsed = time.time() - start_time
            print(f"  step {step+1:4d} | loss {avg_loss:.4f} | lr {lr:.6f} | "
                  f"{elapsed:.1f}s")

    # 返回平均 loss
    return total_loss / max(total_steps, 1)


def evaluate(model, dataloader, criterion, device):
    """
    在验证集上评估
    """
    # 切换到评估模式：关闭 dropout
    model.eval()
    total_loss = 0.0
    total_steps = 0
    # 关闭梯度
    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_label = tgt[:, 1:]
            logits = model(src, tgt_input)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_label.reshape(-1)
            )
            total_loss += loss.item()
            total_steps += 1
    return total_loss / max(total_steps, 1)


def main():
    # 1. 选择设备
    device = get_device()
    print(f"使用设备: {device}")

    # 2. 构建数据集
    # 用一个小一点的词表，方便训练
    vocab_size = 20
    train_ds = SyntheticDataset(num_samples=2000, min_len=3, max_len=10,
                                vocab_size=vocab_size, seed=42)
    val_ds = SyntheticDataset(num_samples=200, min_len=3, max_len=10,
                              vocab_size=vocab_size, seed=123)

    # 3. 构建 DataLoader
    train_loader = build_dataloader(train_ds, batch_size=32, shuffle=True,
                                    pad_token_id=config.pad_token_id)
    val_loader = build_dataloader(val_ds, batch_size=32, shuffle=False,
                                  pad_token_id=config.pad_token_id)

    # 4. 构建模型
    # 用小一点的模型，训练快
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=128,        # 比论文的 512 小
        n_layers=2,         # 比论文的 6 小
        n_heads=4,          # 比论文的 8 小
        d_ff=512,           # 比论文的 2048 小
        max_seq_len=32,
        dropout=0.1,
        pad_token_id=config.pad_token_id,
    ).to(device)

    # 5. 统计参数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数量: {num_params:,}")

    # 6. 损失函数
    # ignore_index=pad_token_id：padding 位置不计算损失
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)

    # 7. 优化器：Adam，和论文一致
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,             # 初始 lr 会被 scheduler 覆盖
        betas=(0.9, 0.98),  # 论文里的 beta1, beta2
        eps=1e-9,           # 论文里的 eps
    )

    # 8. 学习率调度器：Noam
    scheduler = NoamScheduler(optimizer, d_model=128, warmup_steps=400)

    # 9. 训练循环
    num_epochs = 5
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")
        # 训练一个 epoch
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, config.pad_token_id, clip_grad=1.0, log_every=20
        )
        # 验证
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
            print(f"  已保存最佳模型 (val loss {val_loss:.4f})")

    # 10. 训练结束，保存最终模型
    torch.save(model.state_dict(), "final_model.pt")
    print("\n训练完成，模型已保存。")


if __name__ == "__main__":
    main()