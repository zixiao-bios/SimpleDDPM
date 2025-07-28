# DDPM 参数
n_steps = 1000
min_beta = 1e-4
max_beta = 0.02

# UNet 参数
pe_dim = 10                     # 位置编码的维度，随后通过线性层映射到通道数，并加到特征图上
channels = [30, 60, 100, 180]   # 编解码器的层数等于列表长度，每层通道数等于对应元素
residual = True

# 训练参数
num_workers = 6
batch_size = 128
lr = 1e-3
epochs = 800

# 数据集参数
img_path = "/workspace/FFHQ-64x64/imgs"
input_shape = (3, 64, 64)       # 输入图像的形状，以使用的数据集为准
