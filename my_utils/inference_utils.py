import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import numpy as np
import matplotlib.pyplot as plt

from utils import inference_utils
from my_utils.cache import Cache

img_size = 256

def direct_forward(model, x):
    return model(x)

def cos_sim(vec1, vec2):
    return vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# def check_cache(vec, cache_list, ratio):
#     # 余弦相似度统计表
#     similarity_dict = []

#     # 标准化参考向量vec，需要是2D向量
#     vec = vec.detach().numpy()

#     # 计算需要处理的键的数量
#     num_keys_to_process = int(len(cache_list) * ratio)

#     # 切片列表以获取需要处理的一部分向量
#     values_to_process = cache_list[:num_keys_to_process]

#     for value in values_to_process:
#         # 标准化cache中的向量
#         # value = F.normalize(value, p=2, dim=-1).unsqueeze(0)

#         similarity = cos_sim(vec, value)
#         similarity_dict.append(similarity.item())

#     # 根据相似度排序并获取对应的键
#     sorted_indices = sorted(range(len(similarity_dict)), key=lambda i: similarity_dict[i], reverse=True)
#     first_index = sorted_indices[0]
#     second_index = sorted_indices[1]

#     score = (similarity_dict[first_index] - similarity_dict[second_index]) / similarity_dict[second_index]
#     if score > 0.5:
#         print("amazing, score is more than 0.5")

#     return None

def check_cache(vec, cache_array, ratio):
    # 余弦相似度统计表
    similarity_dict = []

    # 标准化参考向量vec
    vec = vec.detach().numpy()
    vec = vec / np.linalg.norm(vec)

    # 计算需要处理的键的数量
    num_keys_to_process = int(cache_array.shape[0] * ratio)

    # 切片数组以获取需要处理的一部分向量
    values_to_process = cache_array[:num_keys_to_process, :]
    
    for row in values_to_process:
        # 标准化cache中的向量
        row = row / np.linalg.norm(row)

        similarity = np.dot(vec, row)
        similarity_dict.append(similarity)

    # 根据相似度排序并获取对应的键
    sorted_indices = sorted(range(len(similarity_dict)), key=lambda i: similarity_dict[i], reverse=True)
    first_index = sorted_indices[0]
    second_index = sorted_indices[1]

    score = (similarity_dict[first_index] - similarity_dict[second_index]) / similarity_dict[second_index]
    if score > 0.5:
        print("amazing, score is more than 0.5")

    return None

# def check_cache(vec, cache_array, ratio):
#     # 余弦相似度统计表
#     similarity_dict = []

#     # 标准化参考向量vec
#     vec = vec.detach().numpy()
#     vec = vec / np.linalg.norm(vec)

#     for row in cache_array:
#         # 标准化cache中的向量
#         row = row / np.linalg.norm(row)

#         similarity = np.dot(vec, row)
#         similarity_dict.append(similarity)

#     # 根据相似度排序并获取对应的键
#     sorted_indices = sorted(range(len(similarity_dict)), key=lambda i: similarity_dict[i], reverse=True)
#     first_index = sorted_indices[0]
#     second_index = sorted_indices[1]

#     score = (similarity_dict[first_index] - similarity_dict[second_index]) / similarity_dict[second_index]
#     if score > 0.5:
#         print("amazing, score is more than 0.5")

#     return None


def cached_forward(model_list, cache, gap_layer, x, ratio):
    """
    根据切分好的模型进行带缓存的推理
    """
    
    for idx, sub_model in enumerate(model_list):
        if idx == len(model_list) - 1:
            # x = x.view(x.size(0), 256 * 4 * 4)
            x = torch.flatten(x, 1)
        x = sub_model(x)

        if idx < cache.cache_layer_num and cache.cache_sign_list[idx]:
                # print(idx, sub_model)
                vec = gap_layer(x)
                # print(idx, "before squeeze", vec.shape)
                vec = vec.squeeze()
                hit = check_cache(vec, cache.cache_table[idx], ratio)
        # print(idx, vec.shape)

    return x


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
    for i in range(100):
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
    for idx, (data, labels) in enumerate(data_loader):
        for x in data:
            x = x.unsqueeze(0)
            start_time = time.perf_counter()
            y = direct_forward(model, x)
            end_time = time.perf_counter()

            avg_time += end_time - start_time
        print(f"batch {idx} ended ...")
    print(f"total inference time: {avg_time:.3f} s")
    avg_time /= len(data_loader.dataset)

    print(f"avg inference time: {avg_time * 1000:.3f} ms")

    return avg_time

# 模拟和统计缓存预测的信息 
def cached_infer(model_list, cache, data_loader, device, ratio):
    # 初始化
    for sub_model in model_list:
        sub_model.eval()
        sub_model.to(device)

    # 定义提取中间向量语义表示的 GAP 层
    global_avg_pooling = nn.AdaptiveAvgPool2d(1).to(device)

    # warm up
    warm_up_epoch = 100
    avg_time = 0.0

    print('warm up ...')
    dummy_input = torch.rand(1, 3, img_size, img_size).to(device)
    for i in range(100):
        start_time = time.perf_counter()
        _ = cached_forward(model_list, cache, global_avg_pooling, dummy_input, ratio)
        end_time = time.perf_counter()

        avg_time += end_time - start_time
    avg_time /= warm_up_epoch
    print(f"warm up avg time: {avg_time * 1000:.3f} ms")


    # # 将数据加载器转换成迭代器
    # data_iter = iter(data_loader)

    # # 获取一个批次的数据
    # data, labels = next(data_iter) 

    avg_time = 0.0
    for idx, (data, labels) in enumerate(data_loader):
        for x in data:
            x = x.unsqueeze(0)
            start_time = time.perf_counter()
            y = cached_forward(model_list, cache, global_avg_pooling, x, ratio)
            end_time = time.perf_counter()

            avg_time += end_time - start_time
        print(f"batch {idx} ended ...")
    
    print(f"total inference time: {avg_time:.3f} s")
    avg_time /= len(data_loader.dataset)

    print(f"avg inference time: {avg_time * 1000:.3f} ms")
    return avg_time

# def mono_warm_up_cpu(model, input_data, device, epoch):
#     """ CPU 设备预热"""
#     model.eval()
#     dummy_input = torch.rand(input_data.shape).to(device)
#     with torch.no_grad():
#         for i in range(10):
#             _ = model(dummy_input)

#         avg_time = 0.0
#         for i in range(epoch):
#             start = time.perf_counter()
#             _ = model(dummy_input)
#             end = time.perf_counter()
#             curr_time = end - start
#             avg_time += curr_time
#         avg_time /= epoch
#         print(f"Whole Model CPU Warm Up : {curr_time * 1000:.3f}ms")
#         print("==============================================")




import my_utils.load_data as load_data
import my_utils.load_model as load_model

def test_cache_time(cache_array, ratio, epoch=10000):
    avg_time = 0
    for _ in range(epoch):
        start_time = time.perf_counter()
        check_cache(torch.rand(256), cache_array, ratio)
        end_time = time.perf_counter()
        avg_time += end_time - start_time

    avg_time /= epoch
    print(f"ratio: {ratio}")
    print(f"avg check time: {avg_time * 1000:.3f} ms")
    print(f"avg additional inference time: {avg_time * 1000 * 13:.3f} ms")
    
    return avg_time

if __name__ == "__main__":
    device = "cpu"
    model_type = ["alexnet", "vgg16_bn"]
    cache_file = "/home/wyliang/Neurosurgeon/cache/cache.pkl"
    ratio = 0.1

    # 加载数据集和模型划分
    test_loader = load_data.load_data("imagenet-100")
    model = load_model.load_model(model_type="vgg16_bn")
    sub_models = load_model.model_partition(model, model_type[1])

    # 加载缓存
    cache = Cache()
    cache.random_init()
    # cache.load(cache_file)

    print("use avg_time")
    time_1 = direct_infer(model, test_loader, device)

    print("use cahce ...")
    # time_2 = cached_infer(sub_models, cache, test_loader, device, ratio)


    # #  绘图 （缓存时间测试估计）
    # # 创建数据
    # x = [0.1 * i for i in range(1, 11)]
    # y = []
    # for i in x:
    #     t = test_cache_time(cache.cache_table[5], i, epoch=1000)
    #     y.append(t * 1000)
    # z = [ temp * 13 for temp in y ]

    # # 绘制折线图
    # # 单个缓存查找时间
    # plt.plot(x, y, label="single", color='blue')

    # # 添加数据点的标签
    # for i, j in zip(x, y):
    #     plt.text(i, j, f'{j:.2f}', ha='left', va='bottom')  # 显示每个点的值

    # # 所有缓存时间估计
    # plt.plot(x, z, label="all", color='red')

    # # 添加数据点的标签
    # for i, j in zip(x, z):
    #     plt.text(i, j, f'{j:.2f}', ha='left', va='bottom')  # 显示每个点的值
    
    # # 添加图例
    # plt.legend()

    # # 添加标题和轴标签
    # plt.title('cache look up: relationship of ratio and time cosumed')
    # plt.xlabel('ratio')
    # plt.ylabel('time/ms')

    # # 显示图形
    # plt.savefig("/home/wyliang/Neurosurgeon/figs/ratio_time.png")


    #  绘图 （真实数据集时间测试）
    # 创建数据
    ratios = [0.1 * i for i in range(1, 11)]
    t1_list = [time_1 * 1000] * 10
    t2_list = []
    ext_list = []
    for ratio in ratios:
        print(f"ratio: {ratio}")
        t = cached_infer(sub_models, cache, test_loader, device, ratio)
        t2_list.append(t * 1000)
        ext_list.append((t - time_1) * 1000)

    # 绘制折线图
    # 无缓存 推理时间
    plt.plot(ratios, t1_list, label="without cache", color='red')

    # 添加数据点的标签
    for i, j in zip(ratios, t1_list):
        plt.text(i, j, f'{j:.2f}', ha='left', va='bottom')  # 显示每个点的值

    # 有缓存 推理时间
    plt.plot(ratios, t2_list, label="with cache", color='blue')

    # 添加数据点的标签
    for i, j in zip(ratios, t2_list):
        plt.text(i, j, f'{j:.2f}', ha='left', va='bottom')  # 显示每个点的值
    
    # 额外 推理时间
    plt.plot(ratios, ext_list, label="additional time", color='green')

    # 添加数据点的标签
    for i, j in zip(ratios, ext_list):
        plt.text(i, j, f'{j:.2f}', ha='left', va='bottom')  # 显示每个点的值

    # 添加图例
    plt.legend()

    # 添加标题和轴标签
    plt.title('relationship of cache ratio and inference time')
    plt.xlabel('ratio')
    plt.ylabel('time/ms')

    # 保存图形
    plt.savefig("/home/wyliang/Neurosurgeon/figs/ratio_wholeInferTime.png")

    # print(f"avg normal inference time: {time1 * 1000:.3f} ms")
    # print(f"avg cached inference time: {time2 * 1000:.3f} ms")
    # print(f"avg additional inference time: {(time2 - time1) * 1000:.3f} ms")
    
