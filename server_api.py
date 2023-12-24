import torch
import sys, getopt
from net import net_utils
import warnings
warnings.filterwarnings("ignore")
from net.monitor_server import MonitorServer

import numpy as np

"""
    云端设备api 用于接收客户端上传数据决策缓存，将决策好的缓存发回
    server 启动指令 python server_api.py -i 127.0.0.1 -p 9999 -d cpu -t alex_net
    "-t", "--type"          模型种类参数 "alex_net" "vgg_net" "le_net" "mobile_net"
    "-i", "--ip"            服务端 ip地址
    "-p", "--port"          服务端 开放端口
    "-d", "--device"     是否开启服务端GPU计算 cpu or cuda

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
    ip, port = "127.0.0.1", 8090
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

    # Initialize the CNN model, global class frequency table, global semantic center memory, and semantic center update frequency table;
    # 根据模型种类初始化 全局类频率表，全局语义中心内存（根据某些数据得到），语义中心更新频率表
    # 根据数据集初始化全局类频率表
    data_set = "CIFAR-10"
    if data_set == "CIFAR-10":
        global_freq_table = np.zeros(10)
    global_cache = np.array(range(100000))
    # print(type(global_cache))

    # 需要完成的函数，根据数据集和模型类型获得语义中心全局缓存，全局缓存更新频率表
    # global_memory，update_table = get_memory(data_set, model_type)

    # 开启服务端进行监听
    server_address = (ip, port)
    print('Server started. Listening on', server_address)
    socket_server = net_utils.get_socket_server(ip, port)

    # 循环接受客户端请求并决策缓存
    while True:
        conn, client = net_utils.wait_client(socket_server)

        request_type = net_utils.get_short_data(conn)
        print(f"request \"{request_type}\" successfully received")

        if request_type == "init":
            net_utils.send_short_data(conn, global_freq_table, msg="freq_table")
        elif request_type == "cache request":
            client_freq_table = net_utils.get_short_data(conn)
            print(client_freq_table)
            client_ts_table = net_utils.get_short_data(conn)
            print(client_ts_table)

            # 需要定义的函数，根据必需信息决策缓存分配
            # cache = get_cache(global_cache, client_freq_table, client_ts_table)
            cache = global_cache
            net_utils.send_data(conn, cache, "requested cache")
            


