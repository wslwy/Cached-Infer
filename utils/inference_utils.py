import torch
import torch.nn as nn
import time

from models.LeNet import LeNet
from models.AlexNet import AlexNet
from models.VggNet import vgg16_bn
from models.MobileNet import MobileNet
# from utils.excel_utils import *


def get_dnn_model(model_type: str):
    """
    获取DNN模型
    :param model_type: 模型名字
    :或许还要添加model参数
    :return: 对应的名字
    :从model类定义来看，获得的是随机初始化的模型
    """
    input_channels = 3
    if model_type == "alex_net":
        return AlexNet(input_channels=input_channels, num_classes=10)
    elif model_type == "vgg_net":
        return vgg16_bn(input_channels=input_channels)
    elif model_type == "le_net":
        return LeNet(input_channels=input_channels)
    elif model_type == "mobile_net":
        return MobileNet(input_channels=input_channels)
    else:
        raise RuntimeError("没有对应的DNN模型")



def partition_layer(layer):
    sub_model_list = []

    for block in layer:
        sub_model_list.append(block)

    return sub_model_list

def model_partition(model, index_list, model_type="vgg"):
    """
    model_partition函数可以将一个整体的model,根据缓存所在划分成多个部分
    划分的大致思路：左开右闭，主要划分模型的feature部分
    举例：在第index层之后对模型进行划分
    index = 0 - 代表在初始输入进行划分
    index = 1 - 代表在第1层后对模型进行划分, edge_cloud包括第1层

    :param model: 传入模型
    :param index_list: 模型划分点列表，每个元素表示一个划分点
    :return: 划分之后的子模型列表
    """

    model_list = []
    sub_model = nn.Sequential()
    
    if model_type == "alexNet-own":
        i = 0
        for child in model.children():
            if i == 0:
                j = 0
                idx = 0
                index = index_list[j]
                for layer in child:
                    sub_model.add_module(f"{idx}-{layer.__class__.__name__}", layer)
                    if idx == index:
                        model_list.append(sub_model)
                        sub_model = nn.Sequential()
                        j += 1
                        if j < len(index_list):
                            index = index_list[j]       
                            sub_model = nn.Sequential()
                    idx += 1
                model_list.append(sub_model)
                sub_model = nn.Sequential()
            else:
                sub_model.add_module(f"{idx}-{layer.__class__.__name__}", child) 
            i += 1

        model_list.append(sub_model)
    elif model_type == "vgg":
        # 切分模型的 features 部分
        prev_layer_idx = 0

        for idx in index_list:
            # 创建一个子模型，包括从prev_layer_idx到idx的所有层
            sub_model = nn.Sequential(*list(model.features.children())[prev_layer_idx:idx + 1])
        
            # 添加子模型到ModuleList
            model_list.append(sub_model)
        
            prev_layer_idx = idx + 1

        # 添加 avgpool 层
        model_list[-1].add_module("avgpool", model.avgpool)

        # 添加 classifier 部分
        model_list.append(model.classifier)
    elif model_type == "resnet":

        model_list.append(nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool
        ))

        model_list += partition_layer(model.layer1)
        model_list += partition_layer(model.layer2)
        model_list += partition_layer(model.layer3)
        model_list += partition_layer(model.layer4)

        model_list.append(model.avgpool)

        model_list.append(model.fc)
    elif model_type == "resnet_special":

        model_list.append(nn.Sequential(
            model.conv1_custom,
            model.bn1,
            model.relu,
            model.maxpool
        ))

        model_list += partition_layer(model.layer1)
        model_list += partition_layer(model.layer2)
        model_list += partition_layer(model.layer3)
        model_list += partition_layer(model.layer4)

        model_list.append(model.avgpool)

        model_list.append(model.fc_custom)

    return model_list


def partion_bottleneck(bottleneck):
    sub_model_list = []

    sub_model_list.append(nn.Sequential(
        bottleneck.conv1,
        bottleneck.bn1,
        bottleneck.relu,
    ))

    sub_model_list.append(nn.Sequential(
        bottleneck.conv2,
        bottleneck.bn2,
        bottleneck.relu,
    ))

    sub_model_list.append(nn.Sequential(
        bottleneck.conv3,
        bottleneck.bn3,
    ))

    if bottleneck.downsample is not None:
        sub_model_list.append(
            bottleneck.downsample,
        )
    
    sub_model_list.append(bottleneck.relu)

    return sub_model_list

def partition_basic_block(basic_block):
    sub_model_list = []

    for bottleneck in basic_block:
        sub_model_list += partion_bottleneck(bottleneck)

    return sub_model_list

# 是否添加 model_type参数
def model_partition2(model, index_list, model_type="vgg"):
    """
    model_partition函数可以将一个整体的model,根据缓存所在划分成多个部分
    划分的大致思路：左开右闭，主要划分模型的feature部分
    举例：在第index层之后对模型进行划分
    index = 0 - 代表在初始输入进行划分
    index = 1 - 代表在第1层后对模型进行划分, edge_cloud包括第1层

    :param model: 传入模型
    :param index_list: 模型划分点列表，每个元素表示一个划分点
    :return: 划分之后的子模型列表
    """

    model_list = []
    sub_model = nn.Sequential()
    
    if model_type == "alexNet-own":
        i = 0
        for child in model.children():
            if i == 0:
                j = 0
                idx = 0
                index = index_list[j]
                for layer in child:
                    sub_model.add_module(f"{idx}-{layer.__class__.__name__}", layer)
                    if idx == index:
                        model_list.append(sub_model)
                        sub_model = nn.Sequential()
                        j += 1
                        if j < len(index_list):
                            index = index_list[j]       
                            sub_model = nn.Sequential()
                    idx += 1
                model_list.append(sub_model)
                sub_model = nn.Sequential()
            else:
                sub_model.add_module(f"{idx}-{layer.__class__.__name__}", child) 
            i += 1

        model_list.append(sub_model)
    elif model_type == "vgg":
        # 切分模型的 features 部分
        prev_layer_idx = 0

        for idx in index_list:
            # 创建一个子模型，包括从prev_layer_idx到idx的所有层
            sub_model = nn.Sequential(*list(model.features.children())[prev_layer_idx:idx + 1])
        
            # 添加子模型到ModuleList
            model_list.append(sub_model)
        
            prev_layer_idx = idx + 1

        # 添加 avgpool 层
        model_list[-1].add_module("avgpool", model.avgpool)

        # 添加 classifier 部分
        model_list.append(model.classifier)
    elif model_type == "resnet":

        model_list.append(nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool
        ))

        model_list += partition_basic_block(model.layer1)
        model_list += partition_basic_block(model.layer2)
        model_list += partition_basic_block(model.layer3)
        model_list += partition_basic_block(model.layer4)

        model_list.append(model.avgpool)

        model_list.append(model.fc)
    elif model_type == "resnet_special":

        model_list.append(nn.Sequential(
            model.conv1_custom,
            model.bn1,
            model.relu,
            model.maxpool
        ))

        model_list += partition_basic_block(model.layer1)
        model_list += partition_basic_block(model.layer2)
        model_list += partition_basic_block(model.layer3)
        model_list += partition_basic_block(model.layer4)

        model_list.append(model.avgpool)

        model_list.append(model.fc_custom)

    return model_list

def cached_infer(model_list, cache_list, x, device):
    """
    根据切分好的模型进行带缓存的推理
    """
    start_time = time.perf_counter()
    # 准备数据
    x = x.to(device)

    with torch.no_grad():
        for sub_model in model_list:
            x = sub_model(x)

            # 检查是否需要生成并查找缓存
            # GAP
            # 查找缓存
            # 判断是否命中
    end_time = time.perf_counter()

    all_time = end_time - start_time

    return x, all_time

def show_model_constructor(model,skip=True):
    """
    展示DNN各层结构
    :param model: DNN模型
    :param skip: 是否需要跳过 ReLU BatchNorm Dropout等层
    :return: 展示DNN各层结构
    """
    print("show model constructor as follows: ")
    if len(model) > 0:
        idx = 1
        for layer in model:
            if skip is True:
                if isinstance(layer, nn.ReLU) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.Dropout):
                    continue
            print(f'{idx}-{layer}')
            idx += 1
    else:
        print("this model is a empty model")



def show_features(model, input_data, device, epoch_cpu=50, epoch_gpu=100, skip=True, save=False, sheet_name="model", path=None):
    """
    可以输出DNN各层的性质,并将其保存在excel表格中,输出的主要性质如下：
    ["index", "layerName", "computation_time(ms)", "output_shape", "transport_num", "transport_size(MB)","accumulate_time(ms)"]
    [DNN层下标，层名字，层计算时延，层输出形状，需要传输的浮点数数量，传输大小，从第1层开始的累计推理时延]
    :param model: DNN模型
    :param input_data: 输入数据
    :param device: 指定运行设备
    :param epoch_cpu: cpu循环推理次数
    :param epoch_gpu: gpu循环推理次数
    :param skip: 是否跳过不重要的DNN层
    :param save: 是否将内容保存在excel表格中
    :param sheet_name: excel中的表格名字
    :param path: excel路径
    :return: None
    """
    if device == "cuda":
        if not torch.torch.cuda.is_available():
            raise RuntimeError("运行设备上没有cuda 请调整device参数为cpu")

    # 推理之前对设备进行预热
    warmUp(model, input_data, device)

    if save:
        sheet_name = sheet_name
        value = [["index", "layerName", "computation_time(ms)", "output_shape", "transport_num",
                  "transport_size(MB)", "accumulate_time(ms)"]]
        create_excel_xsl(path, sheet_name, value)


    if len(model) > 0:
        idx = 1
        accumulate_time = 0.0
        for layer in model:
            if skip is True:
                if isinstance(layer, nn.ReLU) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.Dropout):
                    continue

            temp_x = input_data
            # 记录DNN单层的推理时间
            input_data, layer_time = recordTime(layer, temp_x, device, epoch_cpu, epoch_gpu)
            accumulate_time += layer_time

            # 计算中间传输占用大小为多少MB
            total_num = 1
            for num in input_data.shape:
                total_num *= num
            size = total_num * 4 / 1000 / 1000

            print("------------------------------------------------------------------")
            print(f'{idx}-{layer} \n'
                  f'computation time: {layer_time :.3f} ms\n'
                  f'output shape: {input_data.shape}\t transport_num:{total_num}\t transport_size:{size:.3f}MB\t accumulate time:{accumulate_time:.3f}ms\n')

            # 保存到excel表格中
            if save:
                sheet_name = input_data
                value = [[idx, f"{layer}", round(layer_time, 3), f"{input_data.shape}", total_num, round(size, 3),
                          round(accumulate_time, 3)]]
                write_excel_xls_append(path, sheet_name, value)
            idx += 1
        return input_data
    else:
        print("this model is a empty model")
        return input_data



def warmUp(model, input_data, device):
    """
    预热操作：不对设备进行预热的话，收集的数据会有时延偏差
    :param model: DNN模型
    :param input_data: 输入数据
    :param device: 运行设备类型
    :return: None
    """
    epoch = 10
    model = model.to(device)
    for i in range(1):
        if device == "cuda":
            warmUpGpu(model, input_data, device, epoch)
        elif device == "cpu":
            warmUpCpu(model, input_data, device, epoch)


def warmUpGpu(model, input_data, device, epoch):
    """ GPU 设备预热"""
    dummy_input = torch.rand(input_data.shape).to(device)
    with torch.no_grad():
        for i in range(10):
            _ = model(dummy_input)

        avg_time = 0.0
        for i in range(epoch):
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()

            _ = model(dummy_input)

            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            avg_time += curr_time
        avg_time /= epoch
        # print(f"GPU Warm Up : {curr_time:.3f}ms")
        # print("==============================================")


def warmUpCpu(model, input_data, device, epoch):
    """ CPU 设备预热"""
    dummy_input = torch.rand(input_data.shape).to(device)
    with torch.no_grad():
        for i in range(10):
            _ = model(dummy_input)

        avg_time = 0.0
        for i in range(epoch):
            start = time.perf_counter()
            _ = model(dummy_input)
            end = time.perf_counter()
            curr_time = end - start
            avg_time += curr_time
        avg_time /= epoch
        # print(f"CPU Warm Up : {curr_time * 1000:.3f}ms")
        # print("==============================================")



def recordTime(model, input_data, device, epoch_cpu, epoch_gpu):
    """
    记录DNN模型或者DNN层的推理时间 根据设备分发到不同函数上进行计算
    :param model: DNN模型
    :param input_data: 输入数据
    :param device: 运行设备
    :param epoch_cpu: cpu循环推理次数
    :param epoch_gpu: gpu循环推理次数
    :return: 输出结果以及推理时延
    """
    model = model.to(device)
    res_x, computation_time = None, None
    if device == "cuda":
        res_x, computation_time = recordTimeGpu(model, input_data, device, epoch_gpu)
    elif device == "cpu":
        res_x, computation_time = recordTimeCpu(model, input_data, device, epoch_cpu)
    return res_x, computation_time



def recordTimeGpu(model, input_data, device, epoch):
    all_time = 0.0
    with torch.no_grad():
        for i in range(epoch):
            if torch.is_tensor(input_data):
                input_data = torch.rand(input_data.shape).to(device)
            # init loggers
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)

            with torch.no_grad():
                starter.record()
                res_x = model(input_data)
                ender.record()

            # wait for GPU SYNC
            # 关于GPU的计算机制 一定要有下面这一行才能准确测量在GPU上的推理时延
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            all_time += curr_time
        all_time /= epoch
    return res_x, all_time


def recordTimeCpu(model, input_data, device, epoch):
    all_time = 0.0
    for i in range(epoch):
        if torch.is_tensor(input_data):
            input_data = torch.rand(input_data.shape).to(device)

        with torch.no_grad():
            start_time = time.perf_counter()
            res_x = model(input_data)
            end_time = time.perf_counter()

        curr_time = end_time - start_time
        all_time += curr_time
    all_time /= epoch
    return res_x, all_time * 1000
