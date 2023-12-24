import torch
import torch.nn as nn


from my_utils.cache import Cache
import data_pre_utils.load_data_v2 as load_data
import  my_utils.load_model as load_model

import time
import numpy as np
import os
import pickle
import yaml

import copy

img_size = 224
Th = 0.01
W = 60

device = "cpu"

criterion = torch.nn.CrossEntropyLoss().to(device)

def check_cache(vec, cache_array, scores, weight, id2label):
    # 余弦相似度统计表
    similarity_table = np.zeros(len(cache_array), dtype=float)

    # 标准化参考向量vec
    vec = vec.detach().numpy()
    vec = vec / np.linalg.norm(vec)
    
    # 以下两个步骤可以合并
    for idx, row in enumerate(cache_array):
        # 标准化cache中的向量(标准化步骤不一定需要)
        if np.linalg.norm(row) != 0:
            row = row / np.linalg.norm(row)
        else:
            # 在这里处理向量范数为零的情况，可以选择将向量保持为零向量或者采取其他操作
            pass

        similarity = np.dot(vec, row)
        similarity_table[idx] = similarity

    for idx in range(len(scores)):
        scores[idx] += weight * similarity_table[idx]

    # 找到数组中元素的排序索引
    sorted_indices = np.argsort(scores)

    # 最大值的索引是排序后的最后一个元素
    max_index = sorted_indices[-1]

    # 第二大值的索引是排序后的倒数第二个元素
    second_max_index = sorted_indices[-2]

    # 找到最大值和第二大值
    max_score = scores[max_index]
    second_score = scores[second_max_index]

    sep = (max_score - second_score) / second_score

    # cache 匹配信息
    # print(f"{id2label[max_index]}, {id2label[second_max_index]}, {max_score:.5f}, {second_score:.5f}, {sep:.5f}")
    if sep > Th:
        # print("amazing, score is more than Threhold, cache hit")
        # print(f"{id2label[max_index]}, {id2label[second_max_index]}, {max_score:.5f}, {second_score:.5f}, {sep:.5f}")
        hit = 1
    else:
        hit = 0

    return hit, max_index


def cached_forward(model_list, model_type, cache, gap_layer, x, cache_size):
    """
    根据切分好的模型进行带缓存的推理
    """
    # print(f"length: {len(cache.cache_table[0])}")
    hit = 0
    layer_idx = 0
    scores = np.zeros(cache_size, dtype=float)
    for idx, sub_model in enumerate(model_list):
        if idx == len(model_list) - 1:
            x = torch.flatten(x, 1)
        x = sub_model(x)

        if cache_size > 0 and cache.cache_sign_list[idx]:
            vec = gap_layer(x)
            vec = vec.squeeze()
            weight = 1 << layer_idx
            # print(weight, vec)
            hit, pred_id = check_cache(vec, cache.cache_table[idx], scores, weight, cache.id2label)
            if hit:
                x = pred_id
                break
            layer_idx += 1

    return idx, hit, x
    # return layer_idx, hit, x

def select_cache(global_cache, local_cache, cache_size):
    scores = np.zeros(global_cache.cache_size, dtype=float)

    for idx in range(global_cache.cache_size):
        scores[idx] = local_cache.freq_table[idx] * (0.25) ** np.floor(local_cache.ts_table[idx] / W)
    
    local_cache.id2label = np.argsort(scores)[::-1][:cache_size]

    # print(cache_size, len(local_cache.id2label))
    for layer in range(global_cache.cache_layer_num):
        for idx, label in enumerate(local_cache.id2label):
            local_cache.cache_table[layer][idx] = global_cache.cache_table[layer][label]

    # 检查缓存分配结果
    # print(local_cache.id2label, local_cache.cache_table)
    return 

def cached_infer(model_list, model_type, global_cache, local_cache, data_loader, device, cache_size):
    """ 返回平均推理时延 和 准确率"""
    # 初始化
    for sub_model in model_list:
        sub_model.eval()
        sub_model.to(device)

    # 定义提取中间向量语义表示的 GAP 层
    global_avg_pooling = nn.AdaptiveAvgPool2d(1).to(device)

    # warm up
    warm_up_epoch = 100
    total_time = 0.0
    correct = 0

    print('warm up ...')
    dummy_input = torch.rand(1, 3, img_size, img_size).to(device)
    for i in range(warm_up_epoch):
        start_time = time.perf_counter()
        _ = cached_forward(model_list, model_type, local_cache, global_avg_pooling, dummy_input, cache_size)
        end_time = time.perf_counter()

        total_time += end_time - start_time
    avg_time = total_time / warm_up_epoch
    print(f"warm up avg time: {avg_time * 1000:.3f} ms")




    # 将数据加载器转换成迭代器
    # data_iter = iter(data_loader)

    # 获取一个批次的数据
    # data, labels = next(data_iter) 

    total_time = 0.0
    
    for epc, (data, labels) in enumerate(data_loader):

        data = data.to(device)
        labels = labels.to(device)

        select_cache(global_cache, local_cache, cache_size)
        for x, y in zip(data, labels):
            # print(f"label: {y}")
            for idx in range(global_cache.cache_size):
                local_cache.ts_table[idx] += 1
                local_cache.ts_table[y] = 0

            x = x.unsqueeze(0)
            start_time = time.perf_counter()
            hit_idx, hit, res = cached_forward(model_list, model_type, local_cache, global_avg_pooling, x, cache_size)
            end_time = time.perf_counter()

            
            total_time += end_time - start_time

            if hit:
                pred = local_cache.id2label[res]
            else:
                pred = torch.max(res, 1)[1]

            test_correct = (pred == y).sum()
            correct = correct + test_correct.item()
        
        # print(f"ts_table: {local_cache.ts_table}")
        print(f"batch {epc} ended ...")
    

    # 收集指标
    sample_num = len(data_loader.dataset)

    correct_ratio = correct / sample_num

    print(f"total inference time: {total_time:.3f} s")
    avg_time = total_time / sample_num

    print(f"avg inference time: {avg_time * 1000:.3f} ms, correct/total: {correct}/{sample_num}/{correct_ratio}")
    return avg_time * 1000, correct, correct_ratio


if __name__ == "__main__":
    # 加载模型并划分
    device = "cpu"
    dataset_type_list = ["imagenet1k", "imagenet-100", "ucf101"]
    model_type_list = ["vgg16_bn", "resnet50", "resnet101"]
    
    dataset_type = dataset_type_list[2]
    model_type = model_type_list[2]

    # 读取配置文件
    with open('config.yml', 'r') as config_file:
        config = yaml.safe_load(config_file)
        server = config["server"]
        img_dir_list_file = os.path.join(config["datasets"][server]["image_list_dir"], "testlist01.txt")

    batch_size = W
    cache_size = 101

    class_num = 50
    num_per_class = 3

    loaded_model = load_model.load_model(device=device, model_type=model_type, dataset_type=dataset_type)
    # 模型划分
    sub_models = load_model.model_partition(loaded_model, model_type)

    # 加载cache
    if model_type == "vgg16_bn":
        cache_file = "./cache/vgg16_bn-imagenet100.pkl"
    elif model_type == "resnet50":
        cache_file = "./cache/resnet50-imagenet100.pkl"
    elif model_type == "resnet101":
        cache_file = "./cache/resnet101-ucf101.pkl"
    loaded_cache = Cache(state="global", model_type=model_type, data_set=dataset_type, cache_size=101)
    loaded_cache.load(cache_file)

    global_cache = loaded_cache

    # 加载测试数据
    test_loader = load_data.load_data(dataset_type, img_dir_list_file, 64, batch_size, "test", num_per_class, class_num, 5)
    print(len(test_loader))
    print(len(test_loader.dataset))

    # 变化 不同缓存层 与准确率，平均推理时间的验证
    sign_id_lists = [
        [],
        [9],
        [17],
        [25],
        [33],
        [17, 33],
        [25, 33],
        [17, 25],
        [9, 25],
        [17, 25, 33],
        [ 9, 17, 33],
        [ 9, 17, 25, 33],
        [ 5,  9, 13, 17, 21, 25, 29, 33],
        [ 3,  5,  7,  9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33],
        list(range(34))
    ]
    sign_id_list = sign_id_lists[12]


    base_sign_idx_list = []
    for idx, x in enumerate(global_cache.cache_sign_list):
        if x == 1:
            base_sign_idx_list.append(idx)
    # print(base_sign_idx_list)

    sign_list = [0] * len(global_cache.cache_sign_list)
    for idx in sign_id_list:
        sign_list[base_sign_idx_list[idx]] = idx + 1  
    
    # print(sign_list)

    # 缓存推理
    cache_sizes = [int(0.1 * i * 101) for i in range(0, 11)]
    avg_time_list = []
    corrects = []
    correct_ratio_list = []
    for idx, cache_size in enumerate(cache_sizes):
        local_cache = Cache(state="local", model_type=model_type, data_set=dataset_type, cache_size=101)
        local_cache.freq_table = copy.deepcopy(global_cache.freq_table)
        local_cache.cache_sign_list = sign_list

        print(f"idx: {idx}, cache_size : {cache_size} test start ...")
        for layer in range(global_cache.cache_layer_num):
            dim = len(global_cache.cache_table[layer][0])
            local_cache.cache_table[layer] = np.zeros((cache_size, dim), dtype=float)

        avg_time, correct, correct_ratio = cached_infer(sub_models, model_type, global_cache, local_cache, test_loader, device, cache_size)
        avg_time_list.append(avg_time)
        corrects.append(correct)
        correct_ratio_list.append(correct_ratio)

    # 保存信息
    save_data = {
            "cache_sizes"       : cache_sizes,
            "avg_time_list"     : avg_time_list,
            "corrects"          : corrects,
            "correct_ratio_list": correct_ratio_list,
            "sample_num"        : len(test_loader.dataset)
    }


    # 保存数据到文件
    if model_type == "vgg16_bn":
        # file = "results/_cache_layer_hits_test2.pkl"
        file = "results/vgg16_bn_samll_valid_test.pkl"
    elif model_type == "resnet50":
        file = "results/resnet50_samll_valid_test.pkl"
    elif model_type == "resnet101":
        file = "results/resnet101_test_entries.pkl"


    
    with open(file, 'wb') as fo:
        pickle.dump(save_data, fo)


    print(save_data)