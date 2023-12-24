import torch
import torch.nn as nn

import torchvision as tv
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import Subset, DataLoader
import torchvision.models as models     # 模型都在这里
from my_utils.cache import Cache
# import my_utils.load_data as load_data
import data_pre_utils.load_data_v2 as load_data
import  my_utils.load_model as load_model

import os
import pickle
import yaml
import numpy as np

from utils import inference_utils

# 2.设置GPU和transform 
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = "cpu"

# 定义自己的优化器
criterion = torch.nn.CrossEntropyLoss().to(device)

# model_list前向传播函数
def forward(model_list, x):
    for idx, sub_model in enumerate(model_list):
        if idx == len(model_list) - 1:
            # 线性分类器输入展平
            x = torch.flatten(x, 1)
        x = sub_model(x)
        
    return x

def test_list(model_list, test_loader, device):
    for sub_model in model_list:
        sub_model.eval()

    correct = 0.0
    test_loss = 0.0
    for data, labels in test_loader:

        data = data.to(device)
        labels = labels.to(device)

        test_out = forward(model_list, data)

        loss = criterion(test_out, labels)

        test_loss = test_loss + loss.item()
        pred = torch.max(test_out, 1)[1]
        test_correct = (pred == labels).sum()
        correct = correct + test_correct.item()

    print('Test_loss: {}, Test correct: {}'.format(test_loss / len(test_loader.dataset), correct / len(test_loader.dataset)))
    return test_loss / len(test_loader.dataset), correct / len(test_loader.dataset)


def update_equation(a, freq_a, sum_b, freq_b):
    return (a * freq_a + sum_b) / (freq_a + freq_b)


# 根据数据生成缓存
def generate_cache(model_list, data_loader, device, cache, model_type="vgg16_bn"):
    for sub_model in model_list:
        sub_model.eval()

    correct = 0.0
    test_loss = 0.0

    for data, labels in data_loader:

        data = data.to(device)
        labels = labels.to(device)

        x = data
        for idx, sub_model in enumerate(model_list):
            if idx == len(model_list) - 1:
                x = x.view(x.size(0), 256 * 4 * 4)
            x = sub_model(x)
        test_out = x

        loss = criterion(test_out, labels)

        test_loss = test_loss + loss.item()
        pred = torch.max(test_out, 1)[1]
        test_correct = (pred == labels).sum()
        correct = correct + test_correct.item()

    print('Test_loss: {}, Test correct: {}'.format(test_loss / len(data_loader.dataset), correct / len(data_loader.dataset)))
    return test_loss / len(data_loader.dataset), correct / len(data_loader.dataset)


# 根据数据生成缓存
def test_generate_cache(model_list, data_loader, device, cache):
    for sub_model in model_list:
        sub_model.eval()
        sub_model.to(device)

    correct = 0.0
    test_loss = 0.0

    # # 将数据加载器转换成迭代器
    # data_iter = iter(data_loader)

    # # 获取一个批次的数据
    # data, labels = next(data_iter)  
    
    # 使用 nn.AdaptiveAvgPool2d 进行全局平均池化
    global_avg_pooling = nn.AdaptiveAvgPool2d(1)  # 输出的大小为 (1, 1)

    # 批次化处理：
    print(f"dataloader length: {len(data_loader)} ----")

    for batch_id, (data, labels) in enumerate(data_loader):

        data = data.to(device)
        labels = labels.to(device)

        up_data = [labels,]
        x = data

        # forward 部分
        with torch.no_grad():
            # 使用 nn.AdaptiveAvgPool2d 进行全局平均池化
            # global_avg_pooling = nn.AdaptiveAvgPool2d(1)  # 输出的大小为 (1, 1)
            for idx, sub_model in enumerate(model_list):
                if idx == len(model_list) - 1:
                    x = torch.flatten(x, 1)
                x = sub_model(x)

                if cache.cache_sign_list[idx]:
                    # print(idx, sub_model)
                    y = global_avg_pooling(x)
                    y = y.squeeze()
                    up_data.append(y)
            
            test_out = x

        # 统计一个批次所有更新的中间向量
        for _, data in enumerate(zip(*up_data)):
            label, data_tuple = data[0].item(), data[1:]
            for idx, vec in enumerate(data_tuple):
                # 向量标准化
                vec = vec.numpy()
                vec = vec / np.linalg.norm(vec)
                try:
                    cache.up_cache_table[idx][label] += vec
                except:
                    cache.up_cache_table[idx][label] = vec

        # print(cache.up_cache_table[0][43])
        # print(labels)
        # 使用 torch.bincount 统计各个 label 的数量
        label_counts = torch.bincount(labels, minlength=cache.cache_size)
        # 输出统计结果
        # print(label_counts, label_counts.shape)

        # 将更新添加到缓存和频率表上
        for label, label_count in enumerate(label_counts):
            label_count = label_count.item()
            if label_count != 0:
                cache.freq_table[label] += label_count
                for idx in range(len(cache.cache_table)):
                    cache.cache_table[idx][label] = update_equation(cache.cache_table[idx][label], cache.up_freq_table[idx][label], cache.up_cache_table[idx][label], label_count)
                    cache.up_freq_table[idx][label] += label_count
            # print(label, label_count, cache.up_cache_table[0][label], cache.cache_table[0][label])
        cache.update_table_clear()
        # print(cache.cache_table[0][43])
        # print(cache.up_cache_table[0][43])

        loss = criterion(test_out, labels)

        test_loss = test_loss + loss.item()
        pred = torch.max(test_out, 1)[1]
        test_correct = (pred == labels).sum()
        correct = correct + test_correct.item()

        print(f"batch {batch_id} ended ...")
    

    # 测试频率是否正确
    sum1 = 0
    sum2 = 0
    for label in range(cache.cache_size):
        print(label, cache.up_freq_table[0][label], cache.freq_table[label])
        sum1 += cache.up_freq_table[0][label]
        sum2 += cache.freq_table[label]
    print(sum1, sum2)


    print('Test_loss: {}, Test correct: {}'.format(test_loss / len(data_loader.dataset), correct / len(data_loader.dataset)))
    # print('Test_loss: {}, Test correct: {}'.format(test_loss / len(data_loader.dataset), correct / len(data_loader.dataset)))
    return test_loss / len(data_loader.dataset), correct / len(data_loader.dataset)

def test_cache(model_list, data_loader, device, cache, model_type="vgg16_bn"):
    for sub_model in model_list:
        sub_model.eval()
        sub_model.to(device)

    correct = 0.0
    test_loss = 0.0

    # 将数据加载器转换成迭代器
    data_iter = iter(data_loader)

    # 获取一个批次的数据
    data, labels = next(data_iter)  
    
    # # 批次化处理：
    # for data, labels in data_loader:

    data = data.to(device)
    labels = labels.to(device)

    up_data = [labels,]
    x = data

    # forward 部分
    with torch.no_grad():
        # 使用 nn.AdaptiveAvgPool2d 进行全局平均池化
        global_avg_pooling = nn.AdaptiveAvgPool2d(1)  # 输出的大小为 (1, 1)
        for idx, sub_model in enumerate(model_list):
            if idx == len(model_list) - 1:
                x = torch.flatten(x, 1)
            x = sub_model(x)

            if cache.cache_sign_list[idx]:
                # print(idx, sub_model)
                y = global_avg_pooling(x)
                y = y.squeeze()
                up_data.append(y)
        
        test_out = x

    # 统计一个批次所有更新的中间向量
    for _, data in enumerate(zip(*up_data)):
        label, data_tuple = data[0].item(), data[1:]
        for idx, vec in enumerate(data_tuple):
            # print(label, len(cache.up_cache_table[idx][label]), len(vec))
            # 向量标准化
            vec = vec.numpy()
            vec = vec / np.linalg.norm(vec)
            try:
                cache.up_cache_table[idx][label] += vec
            except:
                cache.up_cache_table[idx][label] = vec
    
    # print(cache.up_cache_table[0][43])
    # print(labels)
    # 使用 torch.bincount 统计各个 label 的数量
    label_counts = torch.bincount(labels, minlength=cache.cache_size)
    # 输出统计结果
    # print(label_counts, label_counts.shape)

    # 将更新添加到缓存和频率表上
    for label, label_count in enumerate(label_counts):
        label_count = label_count.item()
        if label_count != 0:
            cache.freq_table[label] += label_count
            for idx in range(len(cache.cache_table)):
                cache.cache_table[idx][label] = update_equation(cache.cache_table[idx][label], cache.up_freq_table[idx][label], cache.up_cache_table[idx][label], label_count)
                cache.up_freq_table[idx][label] += label_count
        # print(label, label_count, cache.up_cache_table[0][label], cache.cache_table[0][label])
    cache.update_table_clear()
    # 无 shuffle测试
    # print(cache.cache_table[0][43], len(cache.cache_table[0][43]))

    loss = criterion(test_out, labels)

    test_loss = test_loss + loss.item()
    pred = torch.max(test_out, 1)[1]
    test_correct = (pred == labels).sum()
    correct = correct + test_correct.item()

    sum1 = 0
    sum2 = 0
    for label in range(cache.cache_size):
        print(label, cache.up_freq_table[0][label], cache.freq_table[label])
        sum1 += cache.up_freq_table[0][label]
        sum2 += cache.freq_table[label]
    print(sum1, sum2)

    print('Test_loss: {}, Test correct: {}'.format(test_loss / len(labels), correct / len(labels)))
    # print('Test_loss: {}, Test correct: {}'.format(test_loss / len(data_loader.dataset), correct / len(data_loader.dataset)))
    return test_loss / len(data_loader.dataset), correct / len(data_loader.dataset)


if __name__ == "__main__":
    # 加载模型架构并加载权重
    device = "cpu"
    dataset_type_list = ["imagenet1k", "imagenet-100", "ucf101"]
    model_type_list = ["vgg16_bn", "resnet50", "resnet101"]
    
    dataset_type = dataset_type_list[2]
    model_type = model_type_list[0]

    # 读取配置文件
    with open('config.yml', 'r') as config_file:
        config = yaml.safe_load(config_file)
        server = config["server"]
        img_dir_list_file = os.path.join(config["datasets"][server]["image_list_dir"], "trainlist01.txt")

    data_loader = load_data.load_data(dataset_type, img_dir_list_file, 64, 64, "test", 5, 101, 5)
    loaded_model = load_model.load_model(device=device, model_type=model_type, dataset_type=dataset_type)

    print(len(data_loader.dataset))
    # 测试加载模型
    # print(loaded_model)
    # print(loaded_model.children())

    # 模型划分
    sub_models = load_model.model_partition(loaded_model, model_type)

    # 测试划分的模型
    # for idx, sub_model in enumerate(sub_models):
    #     print(idx, sub_model)

    
    cache = Cache(state="global", model_type=model_type, data_set=dataset_type, cache_size=101)
    # # cache.display_info()

    # # test_cache(sub_models, data_loader, device, cache, model_type)
    # test_generate_cache(sub_models, data_loader, device, cache)

    file = "./" + model_type + "-ucf101-large.pkl"
    print(file)
    # cache.save(file)

    loaded_cache = cache
    loaded_cache.load(file)

    sum1 = 0
    sum2 = 0
    for label in range(101):
        # print(label, loaded_cache.up_freq_table[0][label], loaded_cache.freq_table[label])
        sum1 += loaded_cache.up_freq_table[0][label]
        sum2 += loaded_cache.freq_table[label]
    print(len(data_loader.dataset), sum1, sum2)
    # loaded_cache.display_info()















# # 创建完整的数据集对象
# full_dataset = CIFAR100(root='./datasets/CIFAR-100', train=True, download=False)

# # 划分训练集和测试集的索引
# train_indices = range(0, 40000)  # 前40,000个样本用于训练
# test_indices = range(40000, 50000)  # 后10,000个样本用于测试

# # 创建训练集和测试集的子集
# train_dataset = Subset(full_dataset, train_indices)
# test_dataset = Subset(full_dataset, test_indices)

# # 定义额外的转换操作
# additional_transform = transforms.Compose([
#     transforms.RandomRotation(30),
#     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
# ])

# # 在子集上添加额外的转换操作
# train_dataset.dataset.transform = transforms.Compose([
#     train_dataset.dataset.transform,
#     additional_transform
# ])
# test_dataset.dataset.transform = transforms.Compose([
#     test_dataset.dataset.transform,
#     additional_transform
# ])

# # 创建训练集和测试集的数据加载器
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)