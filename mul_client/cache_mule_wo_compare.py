import torch
import torch.nn as nn


from my_utils.cache import Cache
import mul_client.load_data as load_data
import  my_utils.load_model as load_model
from my_utils.mul_exit import MulExit

import time
import numpy as np
import os
import pickle
import yaml

import copy

import logging

# 创建 logger 对象
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 创建文件处理器并设置日志级别
logger_file = "mul_client/logs/detailed_log.log"
file_handler = logging.FileHandler(logger_file)

# 将文件处理器添加到 logger
logger.addHandler(file_handler)

img_size = 224
Th = 0.007
W = 60
filter_time = 0.1

# logger 添加注释信息
logger.info(f"img_size      : {img_size}")
logger.info(f"Threshold     : {Th}")
logger.info(f"W             : {W}")

device = "cpu"

criterion = torch.nn.CrossEntropyLoss().to(device)

# mule 相关参数
exit_layers_lists = [
    [],
    [9],
    [17],
    # [25],
    # [29],
    # [33],
    [17, 33],
    # [25, 33],
    # [17, 25],
    # [9, 25],
    # [17, 25, 33],
    # [ 9, 17, 33],
    # [ 9, 21, 29],
    # [ 9, 17, 25, 33]
]

ths_75 = {
    5: 3.5,
    9: 2.7, 
    13: 3.2,
    17: 3.0,
    21: 2.6,
    25: 1.9,
    29: 1.8,
    33: 1.0
}


logger.info(f"exit_layers_lists : {exit_layers_lists}")
logger.info(f"ths_75            : {ths_75}")


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


def cached_forward(model_list, model_type, cache, gap_layer, x, cache_size, cache_update):
    """
    根据切分好的模型进行带缓存的推理
    """
    # print(f"length: {len(cache.cache_table[0])}")
    hit = 0
    layer_idx = 0
    scores = np.zeros(cache_size, dtype=float)
    up_data = dict()

    for idx, sub_model in enumerate(model_list):
        if idx == len(model_list) - 1:
            x = torch.flatten(x, 1)
        x = sub_model(x)

        if cache.cache_sign_list[idx]:
            vec = gap_layer(x)
            vec = vec.squeeze()
            weight = 1 << layer_idx
            if cache_update:
                tmp = vec.detach().numpy()
                tmp = tmp / np.linalg.norm(tmp) 
                up_data[idx] = tmp

            hit, pred_id = check_cache(vec, cache.cache_table[idx], scores, weight, cache.id2label)
            if hit:
                x = pred_id
                break
            layer_idx += 1

    return idx, hit, x, up_data
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

def update_equation(a, freq_a, sum_b, freq_b):
    return (a * freq_a + sum_b) / (freq_a + freq_b)

def cached_infer(sub_models, model_type, global_cache, client_caches, dataLoaders, device, cache_update, cache_add, cache_size):
    """ 返回平均推理时延 和 准确率 列表"""
    # 初始化
    for sub_model in sub_models:
        sub_model.eval()
        sub_model.to(device)

    # 定义提取中间向量语义表示的 GAP 层
    global_avg_pooling = nn.AdaptiveAvgPool2d(1).to(device)

    # warm up
    warm_up_epoch = 100
    total_time = 0.0

    print('warm up ...')
    logger.info("f{warm up ...}")
    dummy_input = torch.rand(1, 3, img_size, img_size).to(device)
    for _ in range(warm_up_epoch):
        start_time = time.perf_counter()
        _ = cached_forward(sub_models, model_type, client_caches[0], global_avg_pooling, dummy_input, cache_size, cache_update)
        end_time = time.perf_counter()

        total_time += end_time - start_time
    avg_time = total_time / warm_up_epoch
    print(f"warm up avg time: {avg_time * 1000:.3f} ms")
    logger.info(f"warm up avg time: {avg_time * 1000:.3f} ms")




    # 将数据加载器转换成迭代器
    data_iters = []
    for data_loader in dataLoaders:
        data_iters.append( iter(data_loader) )

    work_num = len(data_iters)
    iter_signs = [True] * len(data_iters)
    total_times = [0.0] * len(data_iters)
    corrects = [0] * len(data_iters)
    filtered_nums = [0] * len(data_iters)
    
    o_cache_size = cache_size
    epc = -1
    while work_num:
        epc += 1
        logger.info(f"batch: {epc:<4} begin:")
        for cnum, (iter_sign, data_iter) in enumerate(zip(iter_signs, data_iters)):
            if iter_sign:
                # print(cnum, iter_sign)
                try:
                # if True:
                    data, labels = next(data_iter) 

                    data = data.to(device)
                    labels = labels.to(device)

                    cache_size = o_cache_size
                    select_cache(global_cache, client_caches[cnum], cache_size)
                    logger.info(f"local_cache.labels: {client_caches[cnum].id2label}")
                    
                    for x, y in zip(data, labels):
                        # print(f"label: {y}")
                        for idx in range(global_cache.cache_size):
                            client_caches[cnum].ts_table[idx] += 1
                        client_caches[cnum].ts_table[y] = 0

                        x = x.unsqueeze(0)
                        start_time = time.perf_counter()
                        hit_idx, hit, res, up_data = cached_forward(sub_models, model_type, client_caches[cnum], global_avg_pooling, x, cache_size, cache_update)
                        end_time = time.perf_counter()

                        
                        sample_time = end_time - start_time
                        

                        if hit:
                            pred = client_caches[cnum].id2label[res]
                        else:
                            pred = torch.max(res, 1)[1]

                        test_correct = (pred == y).sum()

                        if sample_time < filter_time:
                            total_times[cnum] += sample_time
                            corrects[cnum] += test_correct.item()

                            add_str = ""
                        else:
                            filtered_nums[cnum] += 1
                            add_str = "### filtered"
                        
                        logger.info(f"hit: {hit}, hit_layer: {hit_idx:<2}, y: {y:<3}, pred: {pred.item()}, is_correct: {test_correct}, time: {sample_time * 1000:.3f} ms " + add_str)


                        # 缓存更新部分
                        if not hit and cache_update:
                            logger.info(f"cache updated ...")
                            for idx, sign in enumerate(client_caches[cnum].cache_sign_list):
                                if sign:    # 将缓存更新暂存
                                    client_caches[cnum].up_cache_table[idx][pred] += up_data[idx]
                                    client_caches[cnum].up_freq_table[idx][pred] += 1
                        
                        # 将未命中的添加到缓存中
                        if not hit and cache_add and pred not in client_caches[cnum].id2label:
                            client_caches[cnum].id2label = np.append(client_caches[cnum].id2label, pred)
                            cache_size += 1
                            for idx, sign in enumerate(client_caches[cnum].cache_sign_list):
                                if sign:    # 添加新缓存条目
                                    client_caches[cnum].cache_table[idx] = np.vstack((client_caches[cnum].cache_table[idx], up_data[idx]))
                    # 将暂存的缓存写入全局缓存
                    for idx, sign in enumerate(client_caches[cnum].cache_sign_list):
                        if sign:
                            for label in range(global_cache.cache_size):
                                if client_caches[cnum].up_freq_table[idx][pred]:
                                    global_cache.cache_table[idx][label] = update_equation(global_cache.cache_table[idx][label], global_cache.up_freq_table[idx][label], client_caches[cnum].up_cache_table[idx][label], client_caches[cnum].up_freq_table[idx][label])
                                    global_cache.up_freq_table[idx][label] += client_caches[cnum].up_freq_table[idx][label]

                    client_caches[cnum].update_table_clear()

                    # print(f"ts_table: {.ts_table}")
                    print(f"client {cnum} batch {epc} cache size {cache_size} ended ...")
                    logger.info(f"client {cnum} batch {epc} cache size {cache_size} ended ...")
                    # print(f"client {cnum} batch {epc} ended ...")
                except:
                    iter_signs[cnum] = False
                    work_num -= 1

    # 收集指标
    sample_nums = [len(data_loader.dataset) - filtered_nums[idx] for idx, data_loader in enumerate(dataLoaders)]

    correct_ratios = [float(corrects[cnum]) / sample_nums[cnum] for cnum in range(len(corrects))]
    avg_times = [total_times[cnum] / sample_nums[cnum] * 1000 for cnum in range(len(corrects))]

    for cnum in range(len(corrects)):
        print(f"client: {cnum}, correct/total: {corrects[cnum]}/{sample_nums[cnum]}, accuracy: {correct_ratios[cnum]}")
        print(f"avg inference time: {avg_times[cnum]:.3f} ms")

        logger.info(f"client: {cnum}, correct/total: {corrects[cnum]}/{sample_nums[cnum]}, accuracy: {correct_ratios[cnum]}")
        logger.info(f"avg inference time: {avg_times[cnum]:.3f} ms")
    
    return avg_times, corrects, sample_nums, correct_ratios



def exit_pred(net, x, th):
    res = net(x)
    # print(res)
    # res = F.normalize(res, dim=1)
    # res = torch.sigmoid(res)
    # print(res)
    # sum = torch.sum(res.float())

    _, pred = torch.max(res, 1)

    topk_values, topk_indices = torch.topk(res, k=2)
    topk_indices = topk_indices.to("cpu").squeeze()
    topk_values = topk_values.to("cpu").squeeze()
    # score = topk_values[0].item() / topk_values[1].item()
    # pred = topk_indices[0].item()

    # score /= sum
    # print(f"score {score}, sum: {sum}")
    score = topk_values[0].item() - topk_values[1].item()
    # print(topk_values, topk_indices, score)
    # hit = 0 if score < th else 1
    hit = 0 if score < th else 1

    return hit, pred

def mule_forward(model_list, model_type, mul_exits, x, y):
    """
    根据切分好的模型进行多出口的推理
    特地定制不提前退出，只统计命中与否及结果
    """
    # print(f"length: {len(cache.cache_table[0])}")
    hit = 0

    for idx, sub_model in enumerate(model_list):
        if idx == len(model_list) - 1:
            x = torch.flatten(x, 1)
        x = sub_model(x)

        if mul_exits.exit_sign_list[idx] == 1:
            hit, pred = exit_pred(mul_exits.exit_nets[idx], x, mul_exits.exit_ths[idx])
            if hit:
                break

    res = pred if hit else x

    return hit, idx, res

def mule_infer(sub_models, model_type, mul_exits, dataLoaders, device):
    """ 返回平均推理时延 和 准确率 列表"""
    # 初始化
    for sub_model in sub_models:
        sub_model.eval()
        sub_model.to(device)

    # 初始化多出口网络
    for net in mul_exits.exit_nets.values():
        net.eval()
        net.to(device)

    # warm up
    warm_up_epoch = 100
    total_time = 0.0

    print('warm up ...')
    logger.info("f{warm up ...}")
    dummy_input = torch.rand(1, 3, img_size, img_size).to(device)
    for _ in range(warm_up_epoch):
        start_time = time.perf_counter()
        _ = mule_forward(sub_models, model_type, mul_exits, dummy_input, 0)
        end_time = time.perf_counter()

        total_time += end_time - start_time
    avg_time = total_time / warm_up_epoch
    print(f"warm up avg time: {avg_time * 1000:.3f} ms")
    logger.info(f"warm up avg time: {avg_time * 1000:.3f} ms")




    # 将数据加载器转换成迭代器
    data_iters = []
    for data_loader in dataLoaders:
        data_iters.append( iter(data_loader) )

    work_num = len(data_iters)
    iter_signs = [True] * len(data_iters)
    total_times = [0.0] * len(data_iters)
    corrects = [0] * len(data_iters)
    filtered_nums = [0] * len(data_iters)
    
    epc = -1
    while work_num:
        epc += 1
        logger.info(f"batch: {epc:<4} begin:")
        for cnum, (iter_sign, data_iter) in enumerate(zip(iter_signs, data_iters)):
            if iter_sign:
                # print(cnum, iter_sign)
                try:
                # if True:
                    data, labels = next(data_iter) 

                    data = data.to(device)
                    labels = labels.to(device)

                    for x, y in zip(data, labels):
                        # print(f"label: {y}")
                        x = x.unsqueeze(0)
                        start_time = time.perf_counter()
                        hit, hit_idx, res = mule_forward(sub_models, model_type, mul_exits, x, y)
                        end_time = time.perf_counter()
                        
                        sample_time = end_time - start_time

                        if hit:
                            pred = res
                        else:
                            pred = torch.max(res, 1)[1]

                        test_correct = (pred == y).sum()

                        # filter
                        if sample_time < filter_time:
                            total_times[cnum] += sample_time
                            corrects[cnum] += test_correct.item()

                            add_str = ""
                        else:
                            filtered_nums[cnum] += 1
                            add_str = "### filtered"
                        
                        logger.info(f"hit: {hit}, hit_layer: {hit_idx:<2}, y: {y:<3}, pred: {pred.item()}, is_correct: {test_correct}, time: {sample_time * 1000:.3f} ms " + add_str)

                    # print(f"ts_table: {.ts_table}")
                    print(f"client {cnum} batch {epc}/{len(dataLoaders[cnum])} exits: {mul_exits.exit_layers} ended ...")
                    # print(f"client {cnum} batch {epc} ended ...")
                except:
                    iter_signs[cnum] = False
                    work_num -= 1

    # 收集指标
    sample_nums = [len(data_loader.dataset) - filtered_nums[idx] for idx, data_loader in enumerate(dataLoaders)]

    correct_ratios = [float(corrects[cnum]) / sample_nums[cnum] for cnum in range(len(corrects))]
    avg_times = [total_times[cnum] / sample_nums[cnum] * 1000 for cnum in range(len(corrects))]

    print(f"exits: {mul_exits.exit_layers}")
    for cnum in range(len(corrects)):
        print(f"client: {cnum}, correct/total: {corrects[cnum]}/{sample_nums[cnum]}, accuracy: {correct_ratios[cnum]}")
        print(f"avg inference time: {avg_times[cnum]:.3f} ms")

        logger.info(f"client: {cnum}, correct/total: {corrects[cnum]}/{sample_nums[cnum]}, accuracy: {correct_ratios[cnum]}")
        logger.info(f"avg inference time: {avg_times[cnum]:.3f} ms")

    return avg_times, corrects, sample_nums, correct_ratios


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


    # logger 添加注释信息
    logger.info(f"device        : {device}")
    logger.info(f"dataset_type  : {dataset_type}")
    logger.info(f"model_type    : {model_type}")
    logger.info(f"batch_size    : {batch_size}")


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
    test_batch_size = 60
    client_num = 4
    step = 5

    base = 3
    non_iid_ratio = 7
    num_class_matrix = [
        [base*non_iid_ratio] * 10 + [base] * 10 + [base] * 10 + [base] * 10,
        [base] * 10 + [base*non_iid_ratio] * 10 + [base] * 10 + [base] * 10,
        [base] * 10 + [base] * 10 + [base*non_iid_ratio] * 10 + [base] * 10,
        [base] * 10 + [base] * 10 + [base] * 10 + [base*non_iid_ratio] * 10,
    ]


    # logger 添加注释信息
    logger.info(f"test_batch_size   : {test_batch_size}")
    logger.info(f"client_num        : {client_num}")
    logger.info(f"step              : {step}")
    logger.info(f"base              : {base}")
    logger.info(f"non_iid_ratio     : {non_iid_ratio}")


    dataLoaders = load_data.load_data(test_batch_size=test_batch_size, client_num=4, num_class_matrix=num_class_matrix, step=step)
    for idx, dataloader in enumerate(dataLoaders):
        print(idx, len(dataloader), len(dataloader.dataset))
    
        # logger 添加注释信息
        logger.info(f"len(data_loader) : {len(dataloader)}")
        logger.info(f"len(dataset)     : {len(dataloader.dataset)}")

    
    save_datas = []

    if False:
        # 多客户端多出口推理
        nets_dir = "temp"
        for exit_layers in exit_layers_lists:
            print(exit_layers)
            mul_exits = MulExit(state="global", model_type="resnet101", dataset="ucf101", exit_layers=exit_layers, ths=ths_75)
            mul_exits.load_init_weights(nets_dir)

            print(f"{mul_exits.exit_layers} test begin ...")
            logger.info(f"{mul_exits.exit_layers} test begin ...")
            avg_times, corrects, sample_nums, correct_ratios = mule_infer(sub_models, model_type, mul_exits, dataLoaders, device)
            print("test end ...")
            logger.info("test end ...")

            # 保存
            save_data = {
                "flag": "mule",
                "exit_layers": mul_exits.exit_layers,
                "ths": mul_exits.exit_ths,
                "avg_times" : avg_times,
                "corrects": corrects,
                "correct_ratios": correct_ratios,
                "sample_nums": sample_nums
            }
            
            # logger 添加注释信息
            logger.info(f"###########################")
            logger.info(f"flag          : mule")
            logger.info(f"exit_layers   : {mul_exits.exit_layers}")
            logger.info(f"ths           : {mul_exits.exit_ths}")
            logger.info(f"avg_times     : {avg_times}")
            logger.info(f"corrects      : {corrects}")
            logger.info(f"correct_ratios: {correct_ratios}")
            logger.info(f"sample_nums   : {sample_nums}")
            logger.info(f"###########################")

            print(save_data)
            save_datas.append(save_data)

        # 多客户端多出口推理 ##

    # 多客户端缓存推理
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
    # sign_id_list = sign_id_lists[11]
    select_sign_id_lists = [sign_id_lists[0], sign_id_lists[11]]

    # logger
    logger.info(f"select_sign_id_lists  : {select_sign_id_lists}")

    # # 抽出部分缓存选择
    # select_sign_id_lists = []
    # for number in [0, 8, 10, 11, 12]:
    #     select_sign_id_lists.append(sign_id_lists[number])

    base_sign_idx_list = []
    for idx, x in enumerate(global_cache.cache_sign_list):
        if x == 1:
            base_sign_idx_list.append(idx)
    # print(base_sign_idx_list)

    for sign_id_list in select_sign_id_lists:
        # 构造sign_list

        sign_list = [0] * len(global_cache.cache_sign_list)
        for idx in sign_id_list:
            sign_list[base_sign_idx_list[idx]] = idx + 1  

        if sign_id_list == []:
            cache_sizes = [40]
        else:
            cache_sizes = [20, 25, 30, 35, 40]
        
        for cache_size in cache_sizes:

            logger.info(f"sign_id_list  : {sign_id_list}")
            logger.info(f"cache_size    : {cache_size}")

            # 创建多客户端本地缓存
            client_caches = list()
            for _ in range(client_num):
                local_cache = Cache(state="local", model_type=model_type, data_set=dataset_type, cache_size=101)
                local_cache.freq_table = copy.deepcopy(global_cache.freq_table)
                local_cache.cache_sign_list = sign_list

                for layer in range(global_cache.cache_layer_num):
                    dim = len(global_cache.cache_table[layer][0])
                    local_cache.cache_table[layer] = np.zeros((cache_size, dim), dtype=float)

                client_caches.append(local_cache)


            # 多客户端缓存推理
            cache_update = True
            # cache_add = True
            # cache_update = False
            cache_add = False

            logger.info(f"cache_update  : {cache_update}")
            logger.info(f"cache_add     : {cache_add}")

            print(f"cache_id_list: {sign_id_list}, cache_size: {cache_size} test begin ...")
            logger.info(f"cache_id_list: {sign_id_list}, cache_size: {cache_size} test begin ...")
            avg_time_list, correct_list, sample_num_list, correct_ratio_list = cached_infer(sub_models, model_type, global_cache, client_caches, dataLoaders, device, cache_update, cache_add, cache_size)
            print("test end ...")
            logger.info(f"test end ...")

            # 保存信息
            save_data = {
                "flag"              : "cache",
                "cache_update"      : cache_update,
                "cache_add"         : cache_add,
                "cache_size"        : cache_size,
                "cache_sign_id_list": sign_id_list,
                "avg_time_list"     : avg_time_list,
                "correct_list"      : correct_list,
                "sample_num_list"   : sample_num_list,
                "correct_ratio_list": correct_ratio_list
            }


            # logger 添加注释信息
            logger.info(f"###########################")
            logger.info(f"flag              : cache")
            logger.info(f"cache_update      : {cache_update}")
            logger.info(f"cache_add         : {cache_add}")
            logger.info(f"cache_size        : {cache_size}")
            logger.info(f"cache_sign_id_list: {sign_id_list}")
            logger.info(f"avg_time_list     : {avg_time_list}")
            logger.info(f"correct_list      : {correct_list}")
            logger.info(f"sample_num_list   : {sample_num_list}")
            logger.info(f"correct_ratio_list: {correct_ratio_list}")
            logger.info(f"###########################")

            print(save_data)
            save_datas.append(save_data)


    # 保存数据到文件
    if model_type == "vgg16_bn":
        # file = "results/_cache_layer_hits_test2.pkl"
        file = "mul_client/results/VGG16_bn_caup_mule_wo_compare.pkl"
    elif model_type == "resnet50":
        file = "results/resnet50_samll_valid_test.pkl"
    elif model_type == "resnet101":
        file = "mul_client/results/resnet101_caup_mule_wo_compare.pkl"


    
    with open(file, 'wb') as fo:
        pickle.dump(save_datas, fo)

    for save_data in save_datas:
        logger.info(f"###########################")
        logger.info(f"details   : {save_data}")

    print(len(save_datas))