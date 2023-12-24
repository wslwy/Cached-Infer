import torchvision
import os

# 设置保存数据集的目录
data_dir = './datasets/CIFAR-100'

# 创建目录（如果不存在）
os.makedirs(data_dir, exist_ok=True)

# 下载 CIFAR-100 数据集
train_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True)
test_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True)

print("CIFAR-100 数据集下载完成。")