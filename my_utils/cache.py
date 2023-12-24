import torch
# from collections import defaultdict
import numpy as np

import pickle
# import dill   #使用dill或许可以序列化lambda等

class Cache:
    def __init__(self, state="global", model_type="resnet50", data_set="imagenet-100", cache_size=100):
        
        self.state = state   # "global", "local"
        self.model_type = model_type
        self.cache_size = cache_size

        self.freq_table = np.zeros(self.cache_size, dtype=int)
        self.ts_table = np.zeros(self.cache_size, dtype=int)
        
        if self.state == "local":
            self.id2label = np.zeros(self.cache_size, dtype=int)

        self.create_cache_table()
        if model_type == "vgg16_bn":
            self.cache_layer_num = 13
            self.cache_sign_list = np.array([1] * 13 + [0] * 2, dtype=int)
        elif model_type == "resnet50":
            type_list = [1]
            type_list += [1, 1, 0, 0, 1] + [1, 1, 0, 1] * 2
            type_list += [1, 1, 0, 0, 1] + [1, 1, 0, 1] * 3
            type_list += [1, 1, 0, 0, 1] + [1, 1, 0, 1] * 5
            type_list += [1, 1, 0, 0, 1] + [1, 1, 0, 1] * 2
            type_list += [0, 0]
            self.cache_layer_num = 49
            self.cache_sign_list = np.array(type_list, dtype=int)
        elif model_type == "resnet101":
            self.cache_layer_num = 34
            self.cache_sign_list = np.array([1] * self.cache_layer_num + [0] * 2)


    # 为了测时间随便整的
    def random_init(self):
        self.freq_table = np.random.randint(0, 5000, self.cache_size)
        self.ts_table = np.random.randint(0, 5000, self.cache_size)

        self.cache_table = list()
        self.up_freq_table = list()
        self.up_cache_table = list()

        # 根据模型类型设置每层缓存的条目维度
        if self.model_type == "alexnet":
            cache_dims = [64, 192, 384, 256, 256]
        elif self.model_type == "vgg16_bn":
            cache_dims = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
        elif self.model_type == "resnet50":
            cache_dims = [64]
            cache_dims += [64, 64, 256] * 3
            cache_dims += [128, 128, 512] * 4
            cache_dims += [256, 256, 1024] * 6
            cache_dims += [512, 512, 2048] * 3
        elif self.model_type == "resnet101":
            cache_dims = [64] + [256] * 3 + [512] * 4 + [1024] * 23 + [2048] * 3

        for cache_dim in cache_dims:
            self.cache_table.append(np.random.rand(self.cache_size, cache_dim))
            self.up_freq_table.append(np.random.randint(0, 5000, self.cache_size))
            self.up_cache_table.append(np.random.rand(self.cache_size, cache_dim))
     
    def create_cache_table(self):
        """不使用lambda函数, dict也许可以换成list"""
        self.cache_table = list()
        self.up_freq_table = list()
        self.up_cache_table = list()

        # 根据模型类型设置每层缓存的条目维度
        if self.model_type == "alexnet":
            cache_dims = [64, 192, 384, 256, 256]
        elif self.model_type == "vgg16_bn":
            cache_dims = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
        elif self.model_type == "resnet50":
            cache_dims = [64]
            cache_dims += [64, 64, 256] * 3
            cache_dims += [128, 128, 512] * 4
            cache_dims += [256, 256, 1024] * 6
            cache_dims += [512, 512, 2048] * 3
        elif self.model_type == "resnet101":
            cache_dims = [64] + [256] * 3 + [512] * 4 + [1024] * 23 + [2048] * 3
            
        for cache_dim in cache_dims:
            self.cache_table.append(np.zeros((self.cache_size, cache_dim), dtype=float))
            self.up_freq_table.append(np.zeros(self.cache_size, dtype=int))
            self.up_cache_table.append(np.zeros((self.cache_size, cache_dim), dtype=float))
            # print(idx, cache_dim, self.up_cache_table[idx][0].shape)

    def update_table_clear(self):
        """ 清空缓存更新表中的信息 """
        for table in self.up_cache_table:
            table.fill(0)  # 使用 fill 方法将数组中的所有元素设置为零

    # def to_device(self, device):
    #     # 将类的实例中的张量移到指定的GPU设备上
    #     for idx in range(len(self.cache_table)):
    #         for key in self.cache_table[idx]:
    #             self.cache_table[idx][key] = self.cache_table[idx][key].to(device)
    #             self.up_cache_table[idx][key] = self.up_cache_table[idx][key].to(device)

    # def init_device(self, device):
    #     # """ 一个为了方便的初始化函数 """
    #     # for idx in range(len(self.cache_table)):
    #     #     for key in range(100):
    #     #         self.cache_table[idx][key] = self.cache_table[idx][key].to(device)
    #     #         self.up_cache_table[idx][key] = self.up_cache_table[idx][key].to(device)
    #     """ 一个为了方便的初始化函数 """
    #     for idx in range(len(self.cache_table)):
    #         for key in range(100):
    #             try:
    #                 self.cache_table[idx][key] = self.cache_table[idx][key].to(device)
    #                 self.up_cache_table[idx][key] = self.up_cache_table[idx][key].to(device)
    #             except Exception as e:
    #                 print(f"An error occurred while moving tensor to device: {e}")
    #                 print(f"idx={idx}, key={key}, cache_table device={self.cache_table[idx][key].device}, up_cache_table device={self.up_cache_table[idx][key].device}")
                

    def display_info(self):
        # print(f"state: {self.state}")
        # print(f"freq_table: {self.freq_table}")
        # print(f"ts_table: {self.ts_table}")
        # print(f"cache_table: {self.cache_table}")
        # print(f"up_freq_table: {self.up_freq_table}")
        # print(f"up_cache_table: {self.up_cache_table}")
        print(f"state: {self.state}")
        print(f"cache_layer_num: {self.cache_layer_num}")
        print(f"cache_size: {self.cache_size}")
        print(f"freq_table: {self.freq_table}")
        print(f"ts_table: {self.ts_table}")
        print(f"cache_table: {self.cache_table}")
        print(f"up_freq_table: {self.up_freq_table}")
        print(f"up_cache_table: {self.up_cache_table}")

    def save(self, file="cache/cache.pkl"):
        save_data = {
            "state"          : self.state,
            "cache_layer_num": self.cache_layer_num,
            "cache_sign_list": self.cache_sign_list,
            "cache_size"     : self.cache_size,
            "freq_table"     : self.freq_table,
            "ts_table"       : self.ts_table,
            "cache_table"    : self.cache_table,
            "up_freq_table"  : self.up_freq_table,
            "up_cache_table" : self.up_cache_table
        }
        # 保存数据到文件
        with open(file, 'wb') as fo:
            pickle.dump(save_data, fo)

    def load(self, file="cache/cache.pkl"):
        # 从文件加载数据
        with open(file, 'rb') as fi:
            loaded_data = pickle.load(fi)

        self.state          = loaded_data["state"]
        self.cache_layer_num= loaded_data["cache_layer_num"]
        self.cache_sign_list= loaded_data["cache_sign_list"]
        self.cache_size     = loaded_data["cache_size"]
        self.freq_table     = loaded_data["freq_table"]
        self.ts_table       = loaded_data["ts_table"]
        self.cache_table    = loaded_data["cache_table"]
        self.up_freq_table  = loaded_data["up_freq_table"]
        self.up_cache_table = loaded_data["up_cache_table"]

    # 使用dill处理lambda问题 
    # def save(self, file="cache/cache.pkl"):
    #     save_data = {
    #         "state"         : self.state,
    #         "freq_table"    : self.freq_table,
    #         "ts_table"      : self.ts_table,
    #         "cache_table"   : self.cache_table,
    #         "up_freq_table" : self.up_freq_table,
    #         "up_cache_table": self.up_cache_table
    #     }
    #     # 保存数据到文件
    #     with open('data.pkl', 'wb') as fo:
    #         dill.dump(save_data, fo)

    # def load(self, file="cache/cache.pkl"):
    #     # 从文件加载数据
    #     with open('data.pkl', 'rb') as fi:
    #         loaded_data = dill.load(fi)

    #     self.state          = loaded_data["state"]
    #     self.freq_table     = loaded_data["freq_table"]
    #     self.ts_table       = loaded_data["ts_table"]
    #     self.cache_table    = loaded_data["cache_table"]
    #     self.up_freq_table  = loaded_data["up_freq_table"]
    #     self.up_cache_table = loaded_data["up_cache_table"]

if __name__ == "__main__":
    test_cache = Cache()
    test_cache.random_init()
    test_cache.display_info()
