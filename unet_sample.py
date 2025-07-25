import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()

        # Assume d_model is an even number for convenience
        assert d_model % 2 == 0

        pe = torch.zeros(max_seq_len, d_model)
        i_seq = torch.linspace(0, max_seq_len - 1, max_seq_len)
        j_seq = torch.linspace(0, d_model - 2, d_model // 2)
        pos, two_i = torch.meshgrid(i_seq, j_seq)
        pe_2i = torch.sin(pos / 10000**(two_i / d_model))
        pe_2i_1 = torch.cos(pos / 10000**(two_i / d_model))
        pe = torch.stack((pe_2i, pe_2i_1), 2).reshape(max_seq_len, d_model)

        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.embedding.weight.data = pe
        self.embedding.requires_grad_(False)

    def forward(self, t):
        return self.embedding(t)
    

class UnetBlock(nn.Module):

    def __init__(self, shape, in_c, out_c, residual=False):
        super().__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.activation = nn.ReLU()
        self.residual = residual
        if residual:
            if in_c == out_c:
                self.residual_conv = nn.Identity()
            else:
                self.residual_conv = nn.Conv2d(in_c, out_c, 1)

    def forward(self, x):
        out = self.ln(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.residual:
            out += self.residual_conv(x)
        out = self.activation(out)
        return out


class UNet(nn.Module):
    """
    用于扩散模型的UNet网络架构
    
    该UNet模型包含编码器-解码器结构，支持时间步嵌入，常用于DDPM等扩散模型中。
    网络通过下采样和上采样操作处理不同分辨率的特征，并在每一层融入时间步信息。
    
    Args:
        n_steps (int): 扩散过程的时间步数，用于位置编码
        channels (list): 各层的通道数列表，例如[10, 20, 40, 80]
        pe_dim (int): 位置编码的维度，默认为10
        residual (bool): 是否在UNet块中使用残差连接，默认为False
    """

    def __init__(self,
                 n_steps,
                 channels=[10, 20, 40, 80],
                 pe_dim=10,
                 residual=False) -> None:
        super().__init__()
        
        # 获取输入图像的形状 (忽略此函数的具体实现)
        C, H, W = get_img_shape()
        
        # 计算网络层数和各层的空间尺寸
        layers = len(channels)
        heights = [H]  # 各层的高度列表
        widths = [W]   # 各层的宽度列表
        
        # 计算下采样后各层的空间尺寸
        current_h, current_w = H, W
        for _ in range(layers - 1):
            current_h //= 2  # 每次下采样高度减半
            current_w //= 2  # 每次下采样宽度减半
            heights.append(current_h)
            widths.append(current_w)

        # 初始化时间步位置编码
        self.pe = PositionalEncoding(n_steps, pe_dim)

        # 初始化编码器路径的组件
        self.encoders = nn.ModuleList()        # 编码器块列表
        self.pe_linears_en = nn.ModuleList()   # 编码器中的时间步线性层
        self.downs = nn.ModuleList()           # 下采样层列表

        # 初始化解码器路径的组件
        self.decoders = nn.ModuleList()        # 解码器块列表
        self.pe_linears_de = nn.ModuleList()   # 解码器中的时间步线性层
        self.ups = nn.ModuleList()             # 上采样层列表

        # 构建编码器路径
        prev_channel = C
        for i, (channel, height, width) in enumerate(zip(channels[:-1], heights[:-1], widths[:-1])):
            # 时间步嵌入的线性变换层
            self.pe_linears_en.append(
                nn.Sequential(
                    nn.Linear(pe_dim, prev_channel), 
                    nn.ReLU(),
                    nn.Linear(prev_channel, prev_channel)
                )
            )
            
            # 编码器块：两个连续的UNet块
            self.encoders.append(
                nn.Sequential(
                    UnetBlock((prev_channel, height, width),
                              prev_channel,
                              channel,
                              residual=residual),
                    UnetBlock((channel, height, width),
                              channel,
                              channel,
                              residual=residual)
                )
            )
            
            # 下采样层：步长为2的卷积，将特征图尺寸减半
            self.downs.append(nn.Conv2d(channel, channel, kernel_size=2, stride=2))
            prev_channel = channel

        # 构建中间层（瓶颈层）
        # 中间层的时间步嵌入
        self.pe_mid = nn.Linear(pe_dim, prev_channel)
        
        # 中间层的UNet块
        mid_channel = channels[-1]
        self.mid = nn.Sequential(
            UnetBlock((prev_channel, heights[-1], widths[-1]),
                      prev_channel,
                      mid_channel,
                      residual=residual),
            UnetBlock((mid_channel, heights[-1], widths[-1]),
                      mid_channel,
                      mid_channel,
                      residual=residual),
        )
        prev_channel = mid_channel

        # 构建解码器路径（与编码器对称）
        for i, (channel, height, width) in enumerate(zip(channels[-2::-1], heights[-2::-1], widths[-2::-1])):
            # 解码器中的时间步嵌入线性层
            self.pe_linears_de.append(nn.Linear(pe_dim, prev_channel))
            
            # 上采样层：转置卷积，将特征图尺寸加倍
            self.ups.append(nn.ConvTranspose2d(prev_channel, channel, kernel_size=2, stride=2))
            
            # 解码器块：处理跳跃连接后的特征
            # 输入通道数为channel*2是因为要拼接编码器的跳跃连接
            self.decoders.append(
                nn.Sequential(
                    UnetBlock((channel * 2, height, width),
                              channel * 2,
                              channel,
                              residual=residual),
                    UnetBlock((channel, height, width),
                              channel,
                              channel,
                              residual=residual)
                )
            )
            prev_channel = channel

        # 输出层：将特征映射回原始通道数
        self.conv_out = nn.Conv2d(prev_channel, C, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t):
        """
        UNet的前向传播
        
        Args:
            x (torch.Tensor): 输入图像张量，形状为 (batch_size, channels, height, width)
            t (torch.Tensor): 时间步张量，形状为 (batch_size,)
            
        Returns:
            torch.Tensor: 输出张量，与输入x具有相同的形状
        """
        batch_size = t.shape[0]
        
        # 对时间步进行位置编码
        t_emb = self.pe(t)  # 形状: (batch_size, pe_dim)
        
        # 编码器路径：逐层下采样并保存跳跃连接
        encoder_outputs = []  # 存储各层编码器输出，用于跳跃连接
        
        for pe_linear, encoder, down in zip(self.pe_linears_en, self.encoders, self.downs):
            # 将时间步嵌入转换为空间形状并加到特征图上
            pe_spatial = pe_linear(t_emb).reshape(batch_size, -1, 1, 1)
            
            # 通过编码器块处理特征（加入时间步信息）
            x = encoder(x + pe_spatial)
            
            # 保存当前层输出用于跳跃连接
            encoder_outputs.append(x)
            
            # 下采样到下一层
            x = down(x)
        
        # 中间层处理
        pe_spatial = self.pe_mid(t_emb).reshape(batch_size, -1, 1, 1)
        x = self.mid(x + pe_spatial)
        
        # 解码器路径：逐层上采样并融合跳跃连接
        for pe_linear, decoder, up, encoder_out in zip(
            self.pe_linears_de, self.decoders, self.ups, encoder_outputs[::-1]
        ):
            # 时间步嵌入
            pe_spatial = pe_linear(t_emb).reshape(batch_size, -1, 1, 1)
            
            # 上采样
            x = up(x)

            # 处理尺寸不匹配问题（由于下采样和上采样可能导致的尺寸差异）
            pad_h = encoder_out.shape[2] - x.shape[2]  # 高度差异
            pad_w = encoder_out.shape[3] - x.shape[3]  # 宽度差异
            
            if pad_h > 0 or pad_w > 0:
                # 对上采样后的特征进行填充以匹配编码器输出的尺寸
                x = F.pad(x, (
                    pad_w // 2, pad_w - pad_w // 2,  # 宽度填充
                    pad_h // 2, pad_h - pad_h // 2   # 高度填充
                ))
            
            # 跳跃连接：拼接编码器输出和当前特征
            x = torch.cat((encoder_out, x), dim=1)
            
            # 通过解码器块处理拼接后的特征
            x = decoder(x + pe_spatial)
        
        # 输出层：生成最终预测
        x = self.conv_out(x)
        return x