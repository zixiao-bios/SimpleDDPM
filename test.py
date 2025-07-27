import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from unet import UNet
from config import *
from ddpm import DDPM
from utils import dynamic_normalize, show_step_imgs

weight_path = "weights/unet_weights_420.pth"


def generate_image(ddpm, net, device):
    """生成一张新图片"""
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
        return False
    
    # 显示生成过程的步骤图像
    show_step_imgs(step_imgs)
    return True


def main():
    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"使用设备: {device}")
    
    # 初始化模型
    ddpm = DDPM(n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)
    net = UNet(n_steps=n_steps, pe_dim=pe_dim, input_shape=input_shape, channels=channels, residual=residual, device=device)
    net.load_state_dict(torch.load(weight_path, map_location=device))
    net = net.to(device)
    net.eval()
    
    print("=== DDPM 图像生成器 ===")
    print("按回车键生成新图片，输入 'q' 或 'quit' 退出程序")
    print("=" * 40)
    
    # 生成第一张图片
    print("\n正在生成第一张图片...")
    generate_image(ddpm, net, device)
    
    # 交互循环
    try:
        while True:
            try:
                user_input = input("\n按回车键生成新图片 (输入 'q' 退出): ").strip().lower()
                
                if user_input in ['q', 'quit', 'exit']:
                    print("程序已退出！")
                    break
                elif user_input == '':
                    print("\n正在生成新图片...")
                    success = generate_image(ddpm, net, device)
                    if not success:
                        print("生成失败，请重试")
                else:
                    print("无效输入，请按回车键生成图片或输入 'q' 退出")
                    
            except KeyboardInterrupt:
                print("\n\n程序被用户中断，退出中...")
                break
            except Exception as e:
                print(f"发生错误: {e}")
                break
    finally:
        # 程序退出时的清理工作
        plt.close('all')  # 关闭所有图片窗口
        plt.ioff()  # 关闭交互模式
        print("清理完成，程序已安全退出")


if __name__ == "__main__":
    main()
