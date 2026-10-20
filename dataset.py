# dataset.py
# 数据相关：词表 + 数据集 + DataLoader

import torch
from torch.utils.data import Dataset, DataLoader


class Vocabulary:
    """
    最简单的词表：把 token 映射成 id，把 id 映射回 token
    真实项目里通常用 BPE / SentencePiece，这里只是演示
    """
    def __init__(self):
        # token -> id
        self.token2id = {}
        # id -> token
        self.id2token = {}
        # 特殊 token
        self.pad_token = "<pad>"   # padding
        self.bos_token = "<bos>"   # 句子开头
        self.eos_token = "<eos>"   # 句子结尾
        self.unk_token = "<unk>"   # 未知 token

    def build(self, token_lists):
        """
        根据所有句子构建词表
        :param token_lists: list[list[str]]，每句话是一个 token 列表
        """
        # 先加入特殊 token，保证 pad 的 id 是 0
        specials = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        # 依次分配 id
        for tok in specials:
            if tok not in self.token2id:
                idx = len(self.token2id)
                self.token2id[tok] = idx
                self.id2token[idx] = tok

        # 再收集所有普通 token
        for tokens in token_lists:
            for tok in tokens:
                if tok not in self.token2id:
                    idx = len(self.token2id)
                    self.token2id[tok] = idx
                    self.id2token[idx] = tok

    def encode(self, tokens):
        """
        把 token 列表转成 id 列表
        遇到未知 token 用 unk 的 id
        """
        unk_id = self.token2id[self.unk_token]
        return [self.token2id.get(tok, unk_id) for tok in tokens]

    def decode(self, ids):
        """
        把 id 列表转回 token 列表
        """
        return [self.id2token.get(i, self.unk_token) for i in ids]

    def __len__(self):
        # 词表大小
        return len(self.token2id)


class SyntheticDataset(Dataset):
    """
    合成数据集：目标序列 = 源序列反转
    输入是数字字符串，每个字符是一个 token
    例如：src = "1 2 3 4" -> tgt = "4 3 2 1"
    """
    def __init__(self,
                 num_samples: int = 1000,   # 样本数量
                 min_len: int = 3,          # 序列最短长度
                 max_len: int = 10,         # 序列最长长度
                 vocab_size: int = 20,      # 数字范围 1..vocab_size-1
                 seed: int = 42):           # 随机种子，保证可复现
        super().__init__()
        # 保存参数
        self.num_samples = num_samples
        self.min_len = min_len
        self.max_len = max_len
        self.vocab_size = vocab_size
        # 固定随机种子
        torch.manual_seed(seed)
        # 存储所有样本
        self.samples = []
        # 生成样本
        for _ in range(num_samples):
            # 随机长度
            length = torch.randint(min_len, max_len + 1, (1,)).item()
            # 随机生成源序列，值在 1..vocab_size-1 之间
            src = torch.randint(1, vocab_size, (length,)).tolist()
            # 目标序列是源序列反转
            tgt = list(reversed(src))
            # 存起来
            self.samples.append((src, tgt))

    def __len__(self):
        # 返回样本数量
        return self.num_samples

    def __getitem__(self, idx):
        # 取第 idx 个样本
        src, tgt = self.samples[idx]
        # 转成 LongTensor
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def collate_fn(batch, pad_token_id: int = 0):
    """
    把一个 batch 里的不等长序列 padding 成等长
    :param batch: list of (src, tgt)，每个都是 1D LongTensor
    :param pad_token_id: padding 的 id
    :return: (src_padded, tgt_padded)，都是 [B, max_len]
    """
    # 拆成两个列表
    srcs, tgts = zip(*batch)

    def pad_sequences(seqs):
        # 取这一批里最长的长度
        max_len = max(s.size(0) for s in seqs)
        # 准备结果列表
        padded = []
        for s in seqs:
            # 需要补多少个 pad
            pad_len = max_len - s.size(0)
            if pad_len > 0:
                # 在末尾补 pad
                s = torch.cat([s, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
            padded.append(s)
        # 堆成 [B, max_len]
        return torch.stack(padded, dim=0)

    # 分别 padding
    src_padded = pad_sequences(srcs)
    tgt_padded = pad_sequences(tgts)
    return src_padded, tgt_padded


def build_dataloader(dataset: Dataset,
                     batch_size: int = 32,
                     shuffle: bool = True,
                     pad_token_id: int = 0,
                     num_workers: int = 0):
    """
    构建 DataLoader
    :param dataset: Dataset 实例
    :param batch_size: 批大小
    :param shuffle: 是否打乱
    :param pad_token_id: padding id
    :param num_workers: 子进程数，Windows 上建议 0
    :return: DataLoader
    """
    # 用 lambda 把 pad_token_id 传进 collate_fn
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, pad_token_id=pad_token_id),
    )


if __name__ == "__main__":
    # 简单自测
    ds = SyntheticDataset(num_samples=5)
    print("样本数量:", len(ds))
    src, tgt = ds[0]
    print("src:", src.tolist())
    print("tgt:", tgt.tolist())

    loader = build_dataloader(ds, batch_size=2)
    for src_batch, tgt_batch in loader:
        print("src_batch:", src_batch.shape)
        print("tgt_batch:", tgt_batch.shape)
        break