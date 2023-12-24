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
        # 标准化cache中的向量
        row = row / np.linalg.norm(row)

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
        hit = 1
    else:
        hit = 0

    return hit, max_index


def cached_forward(model_list, cache, gap_layer, x, cache_size):
    """
    根据切分好的模型进行带缓存的推理
    """
    # print(f"length: {len(cache.cache_table[0])}")
    scores = np.zeros(cache_size, dtype=float)
    for idx, sub_model in enumerate(model_list):
        if idx == len(model_list) - 1:
            x = torch.flatten(x, 1)
        x = sub_model(x)

        if idx < cache.cache_layer_num and cache.cache_sign_list[idx]:
            vec = gap_layer(x)
            vec = vec.squeeze()
            weight = 1 << idx
            hit, pred_id = check_cache(vec, cache.cache_table[idx], scores, weight, cache.id2label)
            if hit:
                x = pred_id
                break

    return hit, x

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

def cached_infer(model_list, global_cache, local_cache, data_loader, device, cache_size):
    # 初始化
    for sub_model in model_list:
        sub_model.eval()
        sub_model.to(device)

    # 定义提取中间向量语义表示的 GAP 层
    global_avg_pooling = nn.AdaptiveAvgPool2d(1).to(device)

    # warm up
    warm_up_epoch = 100
    total_time = 0.0
    hits = 0
    correct = 0
    test_loss = 0.0

    print('warm up ...')
    # dummy_input = torch.rand(1, 3, img_size, img_size).to(device)
    # for i in range(100):
    #     start_time = time.perf_counter()
    #     _ = cached_forward(model_list, local_cache, global_avg_pooling, dummy_input)
    #     end_time = time.perf_counter()

    #     total_time += end_time - start_time
    # avg_time = total_time / warm_up_epoch
    # print(f"warm up avg time: {avg_time * 1000:.3f} ms")




    # 将数据加载器转换成迭代器
    data_iter = iter(data_loader)

    # 获取一个批次的数据
    # data, labels = next(data_iter) 

    total_time = 0.0
    
    for epc, (data, labels) in enumerate(data_loader):
    # test_epoch = 5
    # for epc in range(test_epoch):
        # data, labels = next(data_iter) 

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
            hit, res = cached_forward(model_list, local_cache, global_avg_pooling, x, cache_size)
            end_time = time.perf_counter()

            
            total_time += end_time - start_time
            hits += hit

            if hit:
                pred = local_cache.id2label[res]
            else:
                # loss 部分是否可以删去
                loss = criterion(res, y.unsqueeze(0))
                test_loss = test_loss + loss.item()

                pred = torch.max(res, 1)[1]

            test_correct = (pred == y).sum()
            correct = correct + test_correct.item()
        
        # print(f"ts_table: {local_cache.ts_table}")
        print(f"batch {epc} ended ...")
    

    # 收集指标
    sample_num = len(data_loader.dataset)
    # sample_num = test_epoch * len(labels)
    hit_ratio = hits / sample_num
    correct_ratio = correct / sample_num
    print(f"hit times: {hits}, hit_ratio: {hit_ratio}")
    print('Test_loss: {}, Test correct: {}'.format(test_loss / sample_num, correct_ratio))
    # print('Test_loss: {}, Test correct: {}'.format(test_loss / len(data_loader.dataset), correct / len(data_loader.dataset)))
    print(f"total inference time: {total_time:.3f} s")
    avg_time = total_time / sample_num
    # avg_time = total_time / len(data_loader.dataset)

    print(f"avg inference time: {avg_time * 1000:.3f} ms")
    return avg_time * 1000, hit_ratio, correct_ratio


if __name__ == "__main__":
    # 加载模型并划分
    device = "cpu"
    model_type="vgg16_bn"
    batch_size = W 
    cache_size = 100

    loaded_model = load_model.load_model(device, model_type)
    sub_models = load_model.model_partition(loaded_model, model_type)

    # 加载cache
    cache_file = "./cache/vgg16_bn-imagenet100.pkl"
    loaded_cache = Cache()
    loaded_cache.load(cache_file)

    global_cache = loaded_cache
    local_cache = Cache(state="local", cache_size=100)
    local_cache.freq_table = global_cache.freq_table
    

    # 加载测试数据
    test_loader = load_data.load_data("imagenet-100", batch_size, 256, "train", 300, 50)
    print(len(test_loader))
    print(len(test_loader.dataset))

    # 变化cache与准确率，平均推理时间的验证
    cache_size_list = list(range(10, 101, 10))
    # print(cache_size_list)
    # 缓存推理
    avg_time_list = []
    hit_ratio_list = []
    correct_ratio_list = []
    for size in cache_size_list:
        print(f"cache size : {size} test start ...")
        for layer in range(global_cache.cache_layer_num):
            dim = len(global_cache.cache_table[layer][0])
            local_cache.cache_table[layer] = np.zeros((size, dim), dtype=float)

        avg_time, hit_ratio, correct_ratio = cached_infer(sub_models, global_cache, local_cache, test_loader, device, size)
        avg_time_list.append(avg_time)
        hit_ratio_list.append(hit_ratio)
        correct_ratio_list.append(correct_ratio)

    # 保存信息
    save_data = {
            "cache_size_list"   : cache_size_list,
            "avg_time_list"     : avg_time_list,
            "hit_ratio_list"    : hit_ratio_list,
            "correct_ratio_list": correct_ratio_list
    }
    # 保存数据到文件
    file = "results/second_valid_small_test.pkl"
    with open(file, 'wb') as fo:
        pickle.dump(save_data, fo)


    print(save_data)