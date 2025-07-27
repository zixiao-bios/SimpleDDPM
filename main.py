import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from ddpm import DDPM
import torch.nn as nn
from tqdm import tqdm
from tensorboardX import SummaryWriter
import time
import os

from unet import UNet
from config import *


def train(dateset, device, net: nn.Module):
    weights_dir = f'weights/{time.strftime("%Y-%m-%d_%H-%M-%S")}'
    os.makedirs(weights_dir, exist_ok=True)
    
    dataloader = DataLoader(dateset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    ddpm = DDPM(n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    # 使用余弦退火调度器 (CosineAnnealingLR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs,           # 余弦周期长度 = 总训练轮数
        eta_min=lr * 0.01,      # 最小学习率 = 初始学习率的1%
        last_epoch=-1           # 从第一个epoch开始
    )
    
    loss_fn = nn.MSELoss()
    
    net = net.to(device)
    net.train()

    writer = SummaryWriter(log_dir='logs')
    
    print(f"开始训练 - 数据集大小: {len(dateset)}, Batch大小: {batch_size}, Workers: {num_workers}")
    print(f"使用余弦退火调度器 - 初始学习率: {lr:.2e}, 最小学习率: {lr * 0.01:.2e}")
    
    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        
        # 创建batch级别的进度条
        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        epoch_start_time = time.time()
        
        for data_batch in batch_pbar:
            img_batch = data_batch[0].to(device)
            
            step = torch.randint(0, n_steps, (img_batch.shape[0], 1), device=device)
            x_t, noise = ddpm.noise_sample(img_batch, step)

            pred_noise = net(x_t, step)
            loss = loss_fn(pred_noise, noise)
            epoch_loss += loss.item()
            batch_count += 1
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 更新batch进度条，显示当前loss和学习率
            current_lr = scheduler.get_last_lr()[0]
            batch_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}', 
                'Avg Loss': f'{epoch_loss/batch_count:.4f}',
                'LR': f'{current_lr:.2e}'
            })
        
        # 计算平均loss并打印epoch结果
        avg_epoch_loss = epoch_loss / len(dataloader)
        epoch_time = time.time() - epoch_start_time
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1}/{epochs} completed - Average Loss: {avg_epoch_loss:.6f} - LR: {current_lr:.2e} - Time: {epoch_time:.2f}s")
        
        # 记录到 tensorboard
        writer.add_scalar('loss', avg_epoch_loss, epoch)
        writer.add_scalar('learning_rate', current_lr, epoch)
        writer.add_scalar('time', epoch_time, epoch)
        
        # 更新学习率
        scheduler.step()
        
        # 每 10 个 epoch 保存权重
        if (epoch + 1) % 2 == 0:
            torch.save(net.state_dict(), f"{weights_dir}/unet_weights_{epoch+1}.pth")
            print(f"Saved weights to {weights_dir}/unet_weights_{epoch+1}.pth")


def main():
    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # 定义图像变换
    transform = transforms.Compose([
        # 像素值归一化到[0, 1]
        transforms.ToTensor(),
        # 归一化到[-1, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = ImageFolder(root=img_path, transform=transform)
    assert input_shape == dataset[0][0].shape, 'input_shape must be the same as the shape of the dataset'
    
    net = UNet(n_steps=n_steps, pe_dim=pe_dim, input_shape=input_shape, channels=channels, residual=residual, device=device)
    print('start training')
    train(dataset, device, net)

if __name__ == "__main__":
    main()
