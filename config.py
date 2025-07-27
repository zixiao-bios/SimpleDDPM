# DDPM 参数
n_steps = 1000
min_beta = 1e-4
max_beta = 0.02

# UNet 参数
pe_dim = 10
channels = [10, 20, 40, 80]
residual = True

# 训练参数
num_workers = 6
batch_size = 128
lr = 1e-3
epochs = 1000

# 数据集参数
img_path = "/workspace/FFHQ-64x64/imgs"
input_shape = (3, 64, 64)
