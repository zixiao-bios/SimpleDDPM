import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    计算正弦位置编码。
    """

    def __init__(self, d_model, max_len, device):
        """
        正弦位置编码类的构造函数

        :param d_model: 模型的维度（通常是嵌入层的大小）
        :param max_len: 最大序列长度
        :param device: 硬件设备设置（如 'cuda' 或 'cpu'）
        """
        super(PositionalEncoding, self).__init__()

        # 创建一个形状与输入矩阵相同的零矩阵（用于与输入矩阵相加）
        self.encoding = torch.zeros(max_len, d_model, device=device)
        
        # 不需要计算梯度，位置编码是固定的
        self.encoding.requires_grad = False

        # 创建一个包含序列位置的张量
        pos = torch.arange(0, max_len, device=device)
        # pos.shape: [max_len]
        
        # 将位置张量转换为二维，表示每个单词的位置
        pos = pos.float().unsqueeze(dim=1)
        # pos.shape: [max_len, 1]

        # 创建一个表示 d_model 中每个维度的索引张量
        k = torch.arange(0, d_model / 2, device=device).float()
        # k.shape: [d_model / 2]
        
        # 计算位置编码的正弦值，考虑到单词的位置
        # 对于偶数位置，使用正弦函数；对于奇数位置，使用余弦函数
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (2 * k / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (2 * k / d_model)))

    def forward(self, t):
        """从指定的时间步 t 获取位置编码

        Args:
            t (_type_): 时间步，[num, 1]

        Returns:
            _type_: 位置编码，[num, d_model]
        """
        assert t.dim() == 2 and t.shape[1] == 1, f'Error! t.shape: {t.shape} !== [num, 1]'
        assert t.max() < self.encoding.shape[0], f'Error! t.max(): {t.max()} >= max_len of positional encoding'
        # 将 t 压缩为一维，然后用作索引
        return self.encoding[t.squeeze(), :]
