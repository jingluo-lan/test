import torch
from torch.utils.data import DataLoader
from cyclegan import Generator as CycleGANGenerator, Discriminator as CycleGANDiscriminator
from unet import UNet
from numpy_dataset import NumpyDataset
from train import train_cycle_gan_unet
import random
import numpy as np

# 设置随机数种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置参数
num_epochs = 100
lr = 0.00001
batch_size = 1
n_classes = 5
save_path = './outputs'  # 模型保存路径
save_frequency = 10  # 模型保存频率

# 数据集路径
root = '/home/user/lanzheng/CycleGAN/CycleGAN_UNet/Combine_Cyc-UNet/datasets'

# 加载数据集
dataset = NumpyDataset(root, transforms_=None, mode='train', n_classes=n_classes)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型
G_A2B = CycleGANGenerator().to(device)
G_B2A = CycleGANGenerator().to(device)  # 添加第二个生成器
D_A = CycleGANDiscriminator().to(device)  # 添加第一个鉴别器
D_B = CycleGANDiscriminator().to(device)  # 添加第二个鉴别器
unet = UNet(n_channels=1, n_classes=n_classes).to(device)  # 确保参数名称与UNet类的构造函数匹配

# 开始训练
train_cycle_gan_unet(G_A2B, D_B, G_B2A, D_A, unet, dataloader, num_epochs=num_epochs, lr=lr, device=device, n_classes=n_classes, save_path=save_path, save_frequency=save_frequency)
