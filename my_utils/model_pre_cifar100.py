# 1.加载必要的包
import os.path
from os import listdir
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, Subset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import torch.nn
import torchvision as tv
import torchvision.models as models     # 模型都在这里
from models.zy_AlexNet import AlexNet2

import logging
import datetime
import matplotlib.pyplot as plt

# logger 设置
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y%m%d%H%M%S")

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("logs/log_" + formatted_time + ".txt")
handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(logging.INFO)

logger.addHandler(handler)
logger.addHandler(console)


# 2.设置GPU和transform 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# normalize = transforms.Normalize(
#    mean=[0.485, 0.456, 0.406],
#    std=[0.229, 0.224, 0.225]
# )
# transform = transforms.Compose([transforms.ToTensor(), normalize])  # 转换

# 训练集的转换
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 随机裁剪到32x32大小
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])  # 标准化
])

# 测试集的转换
test_transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])  # 标准化
])

#### 另一种 transform 实现思路
# 基础 tansform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

# additional_transform
additional_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 随机裁剪到32x32大小
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
])


# 3.数据预处理 （需要对路径等进行处理）
# img_paths：图片路径；img_labels：图片标签；size_of_images：图片大小
# class DogDataset(Dataset):
#     def __init__(self, img_paths, img_labels, size_of_images):
#         self.img_paths = img_paths
#         self.img_labels = img_labels
#         self.size_of_images = size_of_images

#     def __len__(self):
#         return len(self.img_paths)

#     def __getitem__(self, index):
#         PIL_IMAGE = Image.open(self.img_paths[index]).resize(self.size_of_images)
#         TENSOR_IMAGE = transform(PIL_IMAGE)
#         label = self.img_labels[index]
#         return TENSOR_IMAGE, label

# 加载完整的 CIFAR-100 数据集 
full_dataset = tv.datasets.CIFAR100(root='/data/wyliang/datasets/CIFAR-100', train=True, transform=transform, download=False)
print(len(full_dataset))

# 划分训练集和测试集的索引
test_ratio = 0.2
k = int(len(full_dataset) * (1-test_ratio))
train_indices = range(0, k)  # 前80%个样本用于训练
test_indices = range(k, len(full_dataset))  # 后20%个样本用于测试

# 创建训练集和测试集的子集
train_dataset = Subset(full_dataset, train_indices)
test_dataset = Subset(full_dataset, test_indices)

# 训练集添加额外的 transform
train_dataset.dataset.transform = transforms.Compose([
    train_dataset.dataset.transform,
    additional_transform
])

print(len(train_dataset))
print(len(test_dataset))

# 创建训练集和测试集的数据加载器 (num_workers 看看是否需要设为1)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

# # 测试数据集
# # 获取一个批次的数据
# batch = next(iter(test_loader))

# # 解析批次的数据
# images, labels = batch

# # 打印数据的形状
# print("图像数据形状:", images.shape)
# print("标签数据形状:", labels.shape)
# print(images[0][0])
# print(labels)

# dicte = {}
# for i in range(0, 100):
#     dicte[i] = 0
# for data, label in test_loader:
#     for x in label:
#         dicte[x.item()] += 1
# print(dicte)
# for i in range(100):
#     if abs(dicte[i] - 100) >= 20:
#         print(i)
    

# 4.引入模型
# # 引入预训练好的模型模型
# alexnet = models.alexnet(pretrained=True)
# # 修改最后一层全连接层输出的种类：CIFAR-100 的输出为 100
# num_fc = alexnet.classifier[6].in_features
# alexnet.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=100)
# alexnet = alexnet.to(device)
# # 对于模型的每个权重，使其不进行反向传播，即固定参数
# for param in alexnet.parameters():
#     param.requires_grad = False
# # 但是参数全部固定了，也没法进行学习，所以我们不固定最后一层，即全连接层
# for param in alexnet.classifier[6].parameters():
#     param.requires_grad = True
alexnet = AlexNet2(100)
alexnet = alexnet.to(device)
# 定义自己的优化器
criterion = torch.nn.CrossEntropyLoss().to(device)

# 参数设定
lr = 0.1
dacay_rate = 0.99
min_lr = 0.001
momentom = 0.9
weight_deacy = 0.0005

# 学习率设定
# optimizer = torch.optim.AdamW(alexnet.parameters(), lr=0.001, betas = (0.9, 0.999), weight_decay=0.01, amsgrad=False)
optimizer = torch.optim.Adam(alexnet.parameters(), lr=0.001)

# optimizer = optim.SGD(local_model.parameters(), momentum=cfg['momentum'], lr=epoch_lr, weight_decay=cfg['weight_decay'])


# 5.训练模型
def train(epoch, model, train_loader, criterion, optimizer, device, logger):
    model.train()
    epoch_loss = 0.0
    correct = 0.0

    for data, label in train_loader:
        data = data.to(device)
        label = label.to(device)

        train_output = model(data)

        loss = criterion(train_output, label)

        # 统计一些信息
        epoch_loss = epoch_loss + loss
        pred = torch.max(train_output, 1)[1]
        train_correct = (pred == label).sum()
        correct += train_correct.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    logger.info('Epoch: %d, Train_loss: %f, Train correct: %f', epoch, epoch_loss / len(train_dataset), correct / len(train_dataset))
    return epoch_loss / len(train_dataset), correct / len(train_dataset)

# 6.测试模型
def test(model, test_loader, device, logger):
    model.eval()
    list1 = []
    list2 = []

    correct = 0.0
    test_loss = 0.0
    for data, label in test_loader:

        data = data.to(device)
        label = label.to(device)

        test_out = model(data)

        loss = criterion(test_out, label)

        test_loss = test_loss + loss.item()
        pred = torch.max(test_out, 1)[1]
        test_correct = (pred == label).sum()
        correct = correct + test_correct.item()

    logger.info('Test_loss: %f, Test correct: %f', test_loss / len(test_dataset), correct / len(test_dataset))
    return test_loss / len(test_dataset), correct / len(test_dataset)



# print(alexnet)
# # 训练与测试
# logger.info("begin trainning ......")
# epoch = 50
# for n_epoch in range(epoch):
#     train(n_epoch, alexnet, train_loader, criterion, optimizer, device, logger)
#     test(alexnet, test_loader, device, logger)
# logger.info("train end ......")


# #  绘图
# # 创建数据
# x = [1, 2, 3, 4, 5]
# y = [3, 6, 2, 7, 1]

# # 绘制折线图
# plt.plot(x, y)

# # 添加标题和轴标签
# plt.title('Line Plot')
# plt.xlabel('X')
# plt.ylabel('Y')

# # 显示图形
# plt.savefig("figs/fig.png")


# 7.保存模型
PATH = './model_sd/alexnet-CIFAR-100-'+ formatted_time + '.pth'
test(alexnet, test_loader, device, logger)

# # 保存模型架构和权重
# torch.save(alexnet.state_dict(), PATH)

# 加载模型架构并加载权重
PATH = './model_sd/alexnet-CIFAR-100-20230905211646.pth'
loaded_model = AlexNet2(100)  # 这里根据你的模型架构创建一个新的模型实例
loaded_model.load_state_dict(torch.load(PATH))
loaded_model.to(device)
test(loaded_model, test_loader, device, logger)
