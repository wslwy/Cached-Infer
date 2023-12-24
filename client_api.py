import torch
import sys, getopt
from net import net_utils
from utils import inference_utils
from deployment import neuron_surgeon_deployment
import warnings
warnings.filterwarnings("ignore")
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from net.monitor_client import MonitorClient
from multiprocessing import Process
import multiprocessing

import numpy as np
import torchvision.models as models

"""
    边缘设备api，用于启动边缘设备，进行边缘推理，将缓存决策所需信息发送到server
    client 启动指令 python Client_api.py -i 127.0.0.1 -p 9999 -d cpu -t alex_net
    "-t", "--type"          模型种类参数 "alex_net" "vgg_net" "le_net" "mobile_net"
    "-i", "--ip"            服务端 ip地址
    "-p", "--port"          服务端 开放端口
    "-d", "--device"     是否开启客户端GPU计算 cpu or cuda
"""
if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], "t:i:p:d:", ["type=","ip=","port=","device_on="])
    except getopt.GetoptError:
        print("Invalid command-line arguments")
        sys.exit(2)

    # 处理 options中以元组的方式存在(opt,arg)
    data_set = ""
    model_type = ""
    ip, port = "127.0.0.1", 9999
    device = "cpu"
    for opt, arg in opts:
        if opt in ("-t", "--type"):
            model_type = arg
        elif opt in ("-i", "--ip"):
            ip = arg
        elif opt in ("-p", "--port"):
            port = int(arg)
        elif opt in ("-d", "--device"):
            device = arg

    if device == "cuda" and torch.cuda.is_available() == False:
        raise RuntimeError("本机器上不可以使用cuda")
    
    # Initialize local class occurrence frequency table and time-stamp table
    data_set = "CIFAR-10"
    if data_set == "CIFAR-10":
        TimeStamp_table = np.zeros(10)

    # 读取模型
    # model = inference_utils.get_dnn_model(model_type)
    model = models.alexnet(pretrained=True)
    
    # 模型划分
    if model_type == "alex_net":
        # index_list = [2, 5, 8, 10, 12, 22]
        index_list = [1, 4, 7, 9, 11, 21]
        index_list = [1, 4, 7, 9, 11, 20]
    model_list = inference_utils.model_partition2(model, model_type, index_list)
    
    device = "cuda"
    # 将模型的所有参数加载到设备上后，切分的子模型自然也在设备上
    if device != "cpu":
        model.to(device)    
    # for sub_model in model_list:
    #     print(next(model.parameters()).device)
    #     sub_model = sub_model.to(device)
    # print(model_list)
    print("model deliever finished")
    
    # 连接客户端发送初始化信息，接收 类频率表 等
    conn = net_utils.get_socket_client(ip, port)
    net_utils.send_short_data(conn, "init", msg="init msg")

    freq_table = net_utils.get_short_data(conn)
    print(freq_table)

    conn.close()
    print("================= Client Init Finished. ===================")

    # 进行本地推理循环
    # # while True:
    for i in range(5):
        # 连接客户端，发起缓存请求

        conn = net_utils.get_socket_client(ip, port)

        net_utils.send_short_data(conn, "cache request", msg="cache request msg")
        # 上传缓存决策所需信息，如类频率表，时间戳表
        net_utils.send_short_data(conn, freq_table, msg="client freq table msg")
        
        net_utils.send_short_data(conn, TimeStamp_table, msg="client time stamp table msg")
        # 接收缓存
        cache, send_time = net_utils.get_data(conn)

        # print(type(cache))
        # print(cache[:10])
        # print(send_time)

        # 本地推理
        W = 2
        for i in range(W):
            epoch_time = 0.0
            # 获取推理输入数据
            # x = get_data() 
            x = torch.rand(size=(1, 3, 224, 224), requires_grad=False)

            y, infer_time = inference_utils.cached_infer(model_list, cache, x, device)

            print(f"y is {y}")
            epoch_time += infer_time
        print(f"epoch_time = {epoch_time}")
        
        conn.close()
