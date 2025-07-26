import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def dynamic_normalize(img):
    """根据图像中像素值的范围，动态归一化到[0, 1]"""
    return (img - img.min()) / (img.max() - img.min())

def normalize(img):
    """将[-1, 1]范围的图像归一化到[0, 1]"""
    img = torch.clamp(img, -1, 1)
    return (img + 1) / 2

def show_step_imgs(step_imgs):
    """并排显示一系列 step 的图像，用于可视化训练过程

    Args:
        step_imgs (_type_): 时间步数和图像的映射
    """
    assert len(step_imgs) > 0, 'step_imgs must not be empty'
    
    fig, axs = plt.subplots(1, len(step_imgs), figsize=(15, 5), squeeze=False)
    for i, (step, img) in enumerate(step_imgs.items()):
        x_t_pil = transforms.ToPILImage()(normalize(img.squeeze()))
        axs[0, i].imshow(x_t_pil)
        axs[0, i].set_title(f"Step {step}")
        axs[0, i].axis('off')
    plt.show()
