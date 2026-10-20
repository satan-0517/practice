# utils.py
# 通用工具：掩码、学习率调度、设备选择

import math                     # 计算 sqrt 和幂
import torch                    # PyTorch 核心


def create_padding_mask(seq: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """
    创建 padding 掩码：padding 位置为 True，需要被 mask
    :param seq: 输入序列 [B, seq_len]
    :param pad_token_id: padding token 的 id
    :return: [B, 1, 1, seq_len]
    """
    # seq == pad_token_id 得到 [B, seq_len] 的布尔张量
    # 再补两个维度：[B, 1, 1, seq_len]，方便广播到注意力的 scores
    mask = (seq == pad_token_id).unsqueeze(1).unsqueeze(2)
    return mask


def create_look_ahead_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    创建前瞻掩码：上三角为 True，防止解码器看到未来 token
    :param seq_len: 序列长度
    :param device: 放在哪个设备上
    :return: [1, 1, seq_len, seq_len]
    """
    # torch.ones(seq_len, seq_len) 全 1 矩阵
    # triu(diagonal=1) 取上三角（不含对角线）
    # 例如 seq_len=3：
    #   [[0, 1, 1],
    #    [0, 0, 1],
    #    [0, 0, 0]]
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    # 加两个维度 -> [1, 1, seq_len, seq_len]
    mask = mask.unsqueeze(0).unsqueeze(0)
    # 如果指定了 device，就移动过去
    if device is not None:
        mask = mask.to(device)
    return mask


class NoamScheduler:
    """
    论文 5.3 节的学习率调度：
    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    先线性 warmup，到达 warmup_steps 后按 1/sqrt(step) 衰减
    """
    def __init__(self,
                 optimizer: torch.optim.Optimizer,  # 优化器
                 d_model: int,                      # 模型维度
                 warmup_steps: int = 4000):         # warmup 步数
        # 保存优化器
        self.optimizer = optimizer
        # 保存 d_model
        self.d_model = d_model
        # 保存 warmup_steps
        self.warmup_steps = warmup_steps
        # 当前步数，从 0 开始
        self.step_num = 0

    def step(self):
        """每次 optimizer.step() 之后调用一次，更新学习率"""
        # 步数加 1
        self.step_num += 1
        # 按公式计算学习率
        lrate = self._compute_lr()
        # 把学习率写到优化器的 param_group 里
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lrate

    def _compute_lr(self) -> float:
        """按论文公式计算当前学习率"""
        # d_model 的 -0.5 次方，即 1/sqrt(d_model)
        scale = self.d_model ** -0.5
        # 两项取最小
        a = self.step_num ** -0.5                          # 衰减项
        b = self.step_num * (self.warmup_steps ** -1.5)    # warmup 项
        # 返回最终学习率
        return scale * min(a, b)

    def get_last_lr(self) -> float:
        """返回当前学习率，方便打印日志"""
        return self.optimizer.param_groups[0]["lr"]


def get_device() -> torch.device:
    """自动选择设备：有 GPU 用 GPU，否则用 CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon 的 MPS 后端
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")