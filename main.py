import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import FFHQDataset
from ddpm import DDPM
import torch.nn as nn
from tqdm import tqdm

from unet import UNet


num_workers = 10

batch_size = 24
lr = 1e-3
epochs = 100

n_steps = 1000
min_beta = 1e-4
max_beta = 0.02


def dynamic_normalize(img):
    """根据图像中像素值的范围，动态归一化到[0, 1]"""
    return (img - img.min()) / (img.max() - img.min())

def train(dateset, device, net: nn.Module):
    dataloader = DataLoader(dateset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    ddpm = DDPM(n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    net = net.to(device)
    net.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        
        # 创建batch级别的进度条
        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
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
            
            # 更新batch进度条，显示当前loss
            batch_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}', 
                'Avg Loss': f'{epoch_loss/batch_count:.4f}'
            })
        
        # 计算平均loss并打印epoch结果
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} completed - Average Loss: {avg_epoch_loss:.6f}")
        
        # 保存权重
        torch.save(net.state_dict(), f"weights/unet_weights_{epoch+1}.pth")


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
    
    dataset = FFHQDataset(transform=transform)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # ddpm = DDPM(n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)
    
    # 显示加噪声后的图像
    # noise_steps = [0, 10, 50, 100, 200, 500, 999]
    # fig, axs = plt.subplots(batch_size, len(noise_steps), figsize=(15, 5))
    # data_batch = next(iter(dataloader))
    # for i, step in enumerate(noise_steps):
    #     img_batch = data_batch[0].to(device)
    #     step_batch = torch.full((batch_size, 1), step, device=device)
    #     x_t, noise = ddpm.noise_sample(img_batch, step_batch)
    #     for j in range(batch_size):
    #         x_t_pil = transforms.ToPILImage()(dynamic_normalize(x_t[j]))
    #         axs[j, i].imshow(x_t_pil)
    #         axs[j, i].set_title(f"Step {step}")
    #         axs[j, i].axis('off')
    # plt.show()
    
    net = UNet(n_steps=n_steps, pe_dim=10, input_shape=dataset[0][0].shape, device=device)
    print('start training')
    train(dataset, device, net)

if __name__ == "__main__":
    main()
