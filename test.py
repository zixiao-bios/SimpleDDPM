import torch
import torchvision.transforms as transforms

from unet import UNet
from config import *
from ddpm import DDPM
from utils import dynamic_normalize, show_step_imgs

weight_path = "weights/unet_weights_1.pth"


def main():
    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    ddpm = DDPM(n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)
    net = UNet(n_steps=n_steps, pe_dim=pe_dim, input_shape=input_shape, device=device)
    net.load_state_dict(torch.load(weight_path, map_location=device))
    net = net.to(device)
    net.eval()
    
    # 随机生成一个噪声
    noise = torch.randn((1, *input_shape), device=device)
    print(f"输入噪声范围: [{noise.min():.4f}, {noise.max():.4f}]")
    
    extra_steps = [0, 10, 50, 100, 200, 500, 999]
    with torch.no_grad():
        x_0, step_imgs = ddpm.backward_sample(noise, net, extra_steps=extra_steps)
    
    print(f"输出x_0形状: {x_0.shape}")
    print(f"输出x_0范围: [{x_0.min():.4f}, {x_0.max():.4f}]")
    print(f"是否包含nan: {torch.isnan(x_0).any()}")
    print(f"是否包含inf: {torch.isinf(x_0).any()}")
    
    # 检查输出是否有效
    if torch.isnan(x_0).any() or torch.isinf(x_0).any():
        print("警告：输出包含nan或inf值!")
        return
    
    # 将输出从[-1,1]范围转换到[0,1]范围用于显示
    # x_0 = torch.squeeze(x_0)
    # x_0 = torch.clamp(x_0, -1, 1)  # 确保在有效范围内
    # x_0 = (x_0 + 1) / 2  # 从[-1,1]转换到[0,1]
    
    # print(f"归一化后的x_0范围: [{x_0.min():.4f}, {x_0.max():.4f}]")
    
    # x_0_pil = transforms.ToPILImage()(dynamic_normalize(x_0))
    # x_0_pil.show()
    
    show_step_imgs(step_imgs)

if __name__ == "__main__":
    main()
