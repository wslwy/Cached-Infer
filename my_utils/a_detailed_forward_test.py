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
from collections import defaultdict
import logging

# 创建 logger 对象
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 创建文件处理器并设置日志级别
logger_file = "logs/detailed_log.log"
file_handler = logging.FileHandler(logger_file)

# 将文件处理器添加到 logger
logger.addHandler(file_handler)



# 重要参数获取与设置
img_size = 224
# Th = 0.01
Th = 0.006
W = 60
filter_time = 0.1

# logger 添加注释信息
logger.info(f"img_size      : {img_size}")
logger.info(f"Threshold     : {Th}")
logger.info(f"W             : {W}")

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

        if cache.cache_sign_list[idx]:
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
    test_loss = 0.0

    print('warm up ...')
    logger.info("f{warm up ...}")
    dummy_input = torch.rand(1, 3, img_size, img_size).to(device)
    for i in range(warm_up_epoch):
        start_time = time.perf_counter()
        _ = cached_forward(model_list, model_type, local_cache, global_avg_pooling, dummy_input, cache_size)
        end_time = time.perf_counter()

        total_time += end_time - start_time
    avg_time = total_time / warm_up_epoch
    print(f"warm up avg time: {avg_time * 1000:.3f} ms")
    logger.info(f"warm up avg time: {avg_time * 1000:.3f} ms")


    # 正式推理
    total_time = 0.0
    layers_hits = defaultdict(int)
    layers_correct = defaultdict(int)
    layers_sum_time = defaultdict(float)
    filtered_num = 0
    
    for epc, (data, labels) in enumerate(data_loader):
        logger.info(f"batch: {epc:<4} begin:")

        data = data.to(device)
        labels = labels.to(device)

        select_cache(global_cache, local_cache, cache_size)
        logger.info(f"local_cache.labels: {local_cache.id2label}")

        for x, y in zip(data, labels):
            # print(f"label: {y}")
            for idx in range(global_cache.cache_size):
                local_cache.ts_table[idx] += 1
                local_cache.ts_table[y] = 0

            x = x.unsqueeze(0)
            start_time = time.perf_counter()
            hit_idx, hit, res = cached_forward(model_list, model_type, local_cache, global_avg_pooling, x, cache_size)
            end_time = time.perf_counter()

            sample_time = end_time - start_time

            if hit:
                pred = local_cache.id2label[res]
            else:
                # # loss 部分是否可以删去
                # loss = criterion(res, y.unsqueeze(0))
                # test_loss = test_loss + loss.item()

                pred = torch.max(res, 1)[1]

            test_correct = (pred == y).sum().item()

            if sample_time < filter_time:
                total_time += sample_time
                correct = correct + test_correct

                # 添加额外详细记录信息
                if hit:
                    layers_hits[hit_idx] += 1
                    layers_correct[hit_idx] += test_correct
                    layers_sum_time[hit_idx] += sample_time * 1000

                add_str = ""
            else:
                filtered_num += 1
                add_str = "### filtered"

            logger.info(f"hit: {hit}, hit_layer: {hit_idx:<2}, y: {y:<3}, pred: {pred.item()}, is_correct: {test_correct}, time: {sample_time * 1000:.3f} ms " + add_str)

        # print(f"ts_table: {local_cache.ts_table}")
        print(f"batch {epc} ended ...")
        logger.info(f"batch {epc} ended ...")
        # if epc == 0:
        #     break


    logger.info(f"sample/filter/total: {len(data_loader.dataset) - filtered_num:<5} / {filtered_num:<5} / {len(data_loader.dataset):<5}")

    # 收集指标
    sample_num = len(data_loader.dataset) - filtered_num

    correct_ratio = correct / sample_num

    print('Test correct: {}'.format(test_loss / sample_num, correct_ratio))
    # print('Test_loss: {}, Test correct: {}'.format(test_loss / len(data_loader.dataset), correct / len(data_loader.dataset)))
    print(f"total inference time: {total_time:.3f} s")
    avg_time = total_time / sample_num

    print(f"avg inference time: {avg_time * 1000:.3f} ms")

    return avg_time * 1000, correct_ratio, sample_num, (layers_hits, layers_correct, layers_sum_time, correct)


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
    base_cache_size = 100

    class_num = 50
    num_per_class = 10

    # logger 添加注释信息
    logger.info(f"device        : {device}")
    logger.info(f"dataset_type  : {dataset_type}")
    logger.info(f"model_type    : {model_type}")
    logger.info(f"batch_size    : {batch_size}")
    logger.info(f"class_num     : {class_num}")
    logger.info(f"num_per_class : {num_per_class}")


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

    # logger 添加注释信息
    logger.info(f"len(data_loader) : {len(test_loader)}")
    logger.info(f"len(dataset)     : {len(test_loader.dataset)}")

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

    # 测试对比两个
    # sign_id_lists = [ sign_id_lists[0], sign_id_lists[8], sign_id_lists[11], sign_id_lists[12] ]
    sign_id_lists = [ sign_id_lists[11] ]


    # logger 添加注释信息
    logger.info(f"sign_id_lists    : {sign_id_lists}")

    sign_lists = []
    base_sign_idx_list = []
    for idx, x in enumerate(global_cache.cache_sign_list):
        if x == 1:
            base_sign_idx_list.append(idx)
    # print(base_sign_idx_list)

    for sign_id_list in sign_id_lists:
        sign_list = [0] * len(global_cache.cache_sign_list)
        for idx in sign_id_list:
            sign_list[base_sign_idx_list[idx]] = idx + 1
        sign_lists.append(sign_list)    
    
    # print(sign_lists)

    cache_size = 50
    # print(cache_size_list)

    # logger 添加注释信息
    logger.info(f"cache_size       : {cache_size}")

    # 缓存推理
    avg_time_list = []
    correct_ratio_list = []
    for idx, sign_list in enumerate(sign_lists):
        local_cache = Cache(state="local", model_type=model_type, data_set=dataset_type, cache_size=101)
        local_cache.freq_table = copy.deepcopy(global_cache.freq_table)
        local_cache.cache_sign_list = sign_list

        print(f"idx: {idx}, sign list : {sign_id_lists[idx]} test start ...")
        for layer in range(global_cache.cache_layer_num):
            dim = len(global_cache.cache_table[layer][0])
            local_cache.cache_table[layer] = np.zeros((cache_size, dim), dtype=float)

        avg_time, correct_ratio, sample_num, addi_info = cached_infer(sub_models, model_type, global_cache, local_cache, test_loader, device, cache_size)
        avg_time_list.append(avg_time)
        correct_ratio_list.append(correct_ratio)

    # 保存信息
    save_data = {
            "cache_sign_list"   : sign_lists,
            "avg_time_list"     : avg_time_list,
            "correct_ratio_list": correct_ratio_list
    }

    # 总结分析性log
    logger.info(f"avg_time: {avg_time_list[0]:.3f}, accuracy: {correct_ratio_list[0]:.6f}, sample_num: {sample_num:<6}, cache_sing_list: {sign_id_lists}")
    layers_hits, layers_correct, layers_sum_time, total_correct = addi_info
    for key in layers_hits.keys():
        logger.info(f"layer: {key:<3} :")
        logger.info(f"time/total/ratio: {layers_sum_time[key]/layers_hits[key]:.3f} / {avg_time_list[0]:.3f} / {layers_sum_time[key] / layers_hits[key] / avg_time_list[0]:.6f} ")
        logger.info(f"hits/total/ratio: {layers_hits[key]:<6} / {sample_num:<6} / {float(layers_hits[key]) / sample_num:.6f}")
        logger.info(f"accs/hits /ratio: {layers_correct[key]:<6} / {layers_hits[key]:<6} / {float(layers_correct[key]) / layers_hits[key]:.6f}")
        logger.info(f"corr/total/ratio: {layers_correct[key]:<6} / {total_correct:<6} / {float(layers_correct[key]) / total_correct:.6f}")

    # 保存数据到文件
    if model_type == "vgg16_bn":
        # file = "results/_cache_layer_hits_test2.pkl"
        file = "results/vgg16_bn_samll_valid_test.pkl"
    elif model_type == "resnet50":
        file = "results/resnet50_samll_valid_test.pkl"
    elif model_type == "resnet101":
        file = "results/a_detailed_cache_forward_01.pkl"


    
    with open(file, 'wb') as fo:
        pickle.dump(save_data, fo)


    print(save_data)