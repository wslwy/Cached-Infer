import torch
import torch.nn as nn


from my_utils.cache import Cache
# import my_utils.load_data as load_data
import my_utils.load_data_v2 as load_data
import  my_utils.load_model as load_model

import time
import numpy as np
import pickle

img_size = 256
Th = 0.02
W = 60

def direct_forward(model, x):
    return model(x)

# 模拟和统计直接推理的信息
def direct_infer(model, data_loader, device):
    # 初始化
    model.eval()
    model.to(device)

    # warm up
    warm_up_epoch = 100
    avg_time = 0.0

    print('warm up ...')
    dummy_input = torch.rand(1, 3, img_size, img_size).to(device)
    for i in range(warm_up_epoch):
        start_time = time.perf_counter()
        _ = direct_forward(model, dummy_input)
        end_time = time.perf_counter()

        avg_time += end_time - start_time
    avg_time /= warm_up_epoch
    print(f"warm up avg time: {avg_time * 1000:.3f} ms")

    print("begin infer ...")
    # # 将数据加载器转换成迭代器
    # data_iter = iter(data_loader)

    # # 获取一个批次的数据
    # data, labels = next(data_iter) 

    avg_time = 0.0
    correct = 0
    for idx, (data, labels) in enumerate(data_loader):
        for x, y in zip(data, labels):
            x = x.unsqueeze(0)
            start_time = time.perf_counter()
            res = direct_forward(model, x)
            end_time = time.perf_counter()

            pred = torch.max(res, 1)[1]
            test_correct = (pred == y).sum()
            correct = correct + test_correct.item()

            avg_time += end_time - start_time
        print(f"batch {idx} ended ...")
    print(f"total inference time: {avg_time:.3f} s")

    sample_num = len(data_loader.dataset)
    avg_time /= sample_num
    accuracy = correct / sample_num

    print(f"avg inference time: {avg_time * 1000:.3f} ms")
    print(f"accucy: {accuracy:.3f}")

    return avg_time, accuracy


if __name__ == "__main__":
    # 加载模型并划分
    device = "cpu"
    model_type_list = ["vgg16_bn", "resnet50"]
    model_type = model_type_list[1]
    batch_size = W
    cache_size = 100

    loaded_model = load_model.load_model(device, model_type) 

    print(loaded_model)
    
    # 加载测试数据
    test_loader = load_data.load_data("imagenet-100", batch_size, 256, "train", 300, 50)
    print(len(test_loader))
    print(len(test_loader.dataset))


    avg_time = 0.0
    accuracy = 0.0
    avg_time, accuracy = direct_infer(loaded_model, test_loader, device)


    
