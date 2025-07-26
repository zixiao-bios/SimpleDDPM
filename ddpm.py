import torch
import torch.nn as nn

class DDPM():
    def __init__(self, n_steps, min_beta, max_beta, device: torch.device = torch.device('cuda')):
        """创建一个 DDPM 算法实例
        Args:
            n_steps (_type_): 采样时间步数
            min_beta (_type_): 最小 beta
            max_beta (_type_): 最大 beta
        """
        self.n_steps = n_steps
        self.device = device

        # 生成线性变化的 beta
        self.betas = torch.linspace(min_beta, max_beta, n_steps, device=device)
        self.alphas = 1 - self.betas
        # 计算累积乘积
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
    
    def noise_sample(self, x_0, t):
        """DDPM 前向过程，根据 x_0 生成第 t 步的加噪声后的样本
        Args:
            x_0 (_type_): 原始样本, shape: (batch_size, channels, height, width)
            t (_type_): 时间步数, shape: (batch_size, 1)
        Returns:
            x_t (_type_): 第 t 步的加噪声后的样本, shape: (batch_size, channels, height, width)
            noise (_type_): 噪声, shape: (batch_size, channels, height, width)
        """
        t = torch.squeeze(t)
        alpha_bar_t = self.alphas_bar[t].reshape(-1, 1, 1, 1)
        
        noise = torch.randn_like(x_0, device=self.device)
        x_t = alpha_bar_t ** 0.5 * x_0 + (1 - alpha_bar_t) ** 0.5 * noise
        return x_t, noise
    
    def backward_sample_one_step(self, x_t, t, net: nn.Module):
        """DDPM 反向过程，根据 x_t 和 t，生成 x_{t-1}
        Args:
            x_t (_type_): 第 t 步的加噪声后的样本, shape: (batch_size, channels, height, width)
            t (_type_): 时间步数, shape: (batch_size, 1)
            net (_type_): 神经网络模型
        Returns:
            x_{t-1} (_type_): 第 t-1 步的原始样本, shape: (batch_size, channels, height, width)
        """
        assert x_t.shape[0] == t.shape[0], 'x_t and t must have the same batch size'
        
        # 预测噪声
        noise_hat = net(x_t, t)
        
        # 如果 t == 0，则对应位置的 z 为 0，否则为随机噪声
        t = torch.squeeze(t)
        z = torch.randn_like(x_t)
        z[t == 0, ...] = 0
        
        # 计算所需参数
        alpha_t = self.alphas[t].reshape(-1, 1, 1, 1)
        alpha_bar_t = self.alphas_bar[t].reshape(-1, 1, 1, 1)
        beta_t = self.betas[t].reshape(-1, 1, 1, 1)
        
        return (
            (x_t - ((1 - alpha_t) / (1 - alpha_bar_t).sqrt()) * noise_hat)
            / alpha_t.sqrt()
            + beta_t.sqrt() * z
        )
    
    def backward_sample(self, x_T, net: nn.Module, extra_steps = []):
        """DDPM 反向过程，根据 x_T 和 net，生成 x_0

        Args:
            x_T (_type_): x_T 推理时从高斯噪声采样
            net (nn.Module): 预测噪声的神经网络
            extra_steps (_type_, optional): 额外返回的时间步数，用于可视化训练过程. Defaults to [].
        """
        extra_imgs = {}
        
        x_t = x_T
        for t in range(self.n_steps - 1, -1, -1):
            if t in extra_steps:
                extra_imgs[t] = x_t
            t_tensor = torch.full((x_t.shape[0], 1), t, device=self.device)
            x_t = self.backward_sample_one_step(x_t, t_tensor, net)
        
        return x_t, extra_imgs
