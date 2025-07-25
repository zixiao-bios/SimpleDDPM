"""时序 UNet 模型，输入为 x_t 和 t，输出为预测的噪声"""
from torch import nn
import torch
import torch.nn.functional as F

from positional_encoding import PositionalEncoding


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, residual=False):
        super().__init__()
        self.residual = residual
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        
        if residual:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        out = self.seq(x)
        if self.residual:
            out = out + self.residual_conv(x)
        return out

class UNet(nn.Module):
    def __init__(
        self,
        n_steps,
        pe_dim,
        input_shape,
        channels=[10, 20, 40, 80],
        residual=False,
        device: torch.device = torch.device('cuda'),
    ):
        """
        时序 UNet 模型，输入为 x_t 和 t，输出为预测的噪声

        Args:
            n_steps (_type_): _description_
            pe_dim (_type_): _description_
            input_shape (_type_): 输入图像的形状，[C, H, W]
            channels (list, optional): _description_. Defaults to [10, 20, 40, 80].
            device (str, optional): _description_. Defaults to 'cuda'.
        """
        super().__init__()
        
        assert len(input_shape) == 3, f'Error! input_shape: {input_shape} is not a 3D tensor'
        
        self.n_steps = n_steps
        self.pe_dim = pe_dim
        self.device = device
        self.input_shape = input_shape
        self.channels = channels
        self.residual = residual
        
        # 位置编码维度为 pe_dim，后续通过线性层映射到特征图的通道数
        self.pe = PositionalEncoding(pe_dim, n_steps, device)
        
        # 编码器组件
        self.en_blocks = nn.ModuleList()
        self.en_downs = nn.ModuleList()
        self.en_pe = nn.ModuleList()
        
        # 构建编码器组件
        prev_channel = input_shape[0]
        for i in range(len(channels) - 1):
            channel = channels[i]
            
            # 将位置编码映射到特征图的通道数
            self.en_pe.append(nn.Sequential(
                nn.Linear(pe_dim, prev_channel),
                nn.ReLU(),
                nn.Linear(prev_channel, prev_channel),
            ))
            
            # 编码器块
            self.en_blocks.append(nn.Sequential(
                UNetBlock(prev_channel, channel, residual=self.residual),
                UNetBlock(channel, channel, residual=self.residual),
            ))
            
            # 下采样
            self.en_downs.append(nn.Conv2d(channel, channel, kernel_size=2, stride=2))
            
            prev_channel = channel
        
        # neck 组件
        channel = channels[-1]
        self.neck = nn.Sequential(
            UNetBlock(prev_channel, channel, residual=self.residual),
            UNetBlock(channel, channel, residual=self.residual),
        )
        
        # 解码器组件
        self.de_blocks = nn.ModuleList()
        self.de_ups = nn.ModuleList()
        self.de_pe = nn.ModuleList()
        
        # 构建解码器组件
        prev_channel = channel
        for i in range(len(channels) - 2, -1, -1):
            channel = channels[i]
            
            # 上采样
            self.de_ups.append(nn.ConvTranspose2d(prev_channel, channel, kernel_size=2, stride=2))
            
            # 将位置编码映射到特征图的通道数
            self.de_pe.append(nn.Sequential(
                nn.Linear(pe_dim, channel),
                nn.ReLU(),
                nn.Linear(channel, channel),
            ))
            
            # 解码器块
            self.de_blocks.append(nn.Sequential(
                # 输入通道数 * 2 ，因为要拼接编码器的跳跃连接
                UNetBlock(channel * 2, channel, residual=self.residual),
                UNetBlock(channel, channel, residual=self.residual),
            ))
            
            prev_channel = channel
        
        # 输出层
        self.out = nn.Sequential(
            nn.Conv2d(channel, self.input_shape[0], kernel_size=3, padding=1),
        )
        
    def forward(self, x, t):
        """
        UNet的前向传播

        Args:
            x (_type_): 输入图像，[batch_size, C, H, W]
            t (_type_): 时间步，[batch_size, 1]
        """
        assert x.shape[1:] == self.input_shape, f'Error! input shape: {x.shape[1:]} !== model input shape: {self.input_shape}'
        assert t.shape[0] == x.shape[0], f'Error! t.shape[0]: {t.shape[0]} !== x.shape[0]: {x.shape[0]}'
        batch_size = t.shape[0]
        
        # 得到位置编码
        t_emb = self.pe(t)
        
        # 保存每层编码器的输出，用于跳跃连接
        encoder_outputs = []
        
        # 进入编码器
        for en_pe, en_block, en_down in zip(self.en_pe, self.en_blocks, self.en_downs):
            # 将位置编码映射到图像的通道数
            pe = en_pe(t_emb).reshape(batch_size, -1, 1, 1)
            
            # 将位置编码加入到特征图
            x = x + pe
            
            # 编码器处理数据
            x = en_block(x)
            
            # 保存当前层的输出
            encoder_outputs.append(x)
            
            # 下采样
            x = en_down(x)
        
        # 进入 neck
        x = self.neck(x)
        
        # 进入解码器
        for de_up, de_pe, de_block, en_out in zip(self.de_ups, self.de_pe, self.de_blocks, encoder_outputs[::-1]):
            # 上采样
            x = de_up(x)
            
            # 处理尺寸不匹配问题：当原始图像尺寸不是2的幂次时需要padding
            pad_h = en_out.shape[2] - x.shape[2]
            pad_w = en_out.shape[3] - x.shape[3]
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (
                    pad_w // 2,          # 左侧填充
                    pad_w - pad_w // 2,  # 右侧填充
                    pad_h // 2,          # 上侧填充
                    pad_h - pad_h // 2   # 下侧填充
                ))
            
            # 加入位置编码
            pe = de_pe(t_emb).reshape(batch_size, -1, 1, 1)
            x = x + pe
            
            # 拼接编码器对应层的输出
            x = torch.cat((en_out, x), dim=1)
            
            # 解码器处理数据
            x = de_block(x)
        
        # 输出
        x = self.out(x)
        
        return x
