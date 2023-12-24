import torch
import torch.nn as nn
import numpy as np

import os
import pickle

import copy


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

class Exit_Net(nn.Module):
    def __init__(self, in_shape, num_class, model_type="resnet101"):
        in_channel, h, w = in_shape

        super(Exit_Net, self).__init__()
        # self.conv1   = conv3x3(in_planes=in_channel, out_planes=512)
        self.bn1     = nn.BatchNorm2d(in_channel)
        self.relu    = nn.ReLU(inplace=True)

        if model_type == "resnet101":
            Th = 2048
        elif model_type == "vgg16_bn":
            Th = 512

        if in_channel < Th:
            k = s = (h + 1) // 2
            p = 1 if h % 2 else 0
            self.maxpool = nn.MaxPool2d(kernel_size=k, stride=s, padding=p)
        # self.gap = nn.AdaptiveAvgPool2d(1)

            self.fc = nn.Linear(in_channel * 2 * 2, num_class)
        else:
            self.maxpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(in_channel, num_class)


    def custom_initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):  # 对全连接层使用Xavier初始化
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            else:
                # print(f"error, module {m} not found")
                pass

    def forward(self, x):
        # x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # x = self.gap(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x



class MulExit():
    def __init__(self, state="global", model_type="vgg16_bn", dataset="ucf101", exit_layers = [1,3,5,7,9,11,13], ths = {1:1,3:1,5:1,7:1,9:1,11:1,13:1}):
        self.state = state   # "global", "local"
        self.model_type = model_type
        self.dataset = dataset

        self.exit_layers = exit_layers

        if self.model_type == "resnet101":
            self.block_num = 36
            self.in_shape_list = [
                (512, 28, 28),
                (1024, 14, 14),
                (2048, 7, 7),
            ]

            if self.dataset == "ucf101":
                self.class_num = 101
                self.in_shape_idxs = [None] * self.block_num
                # 特殊情况初始化
                self.in_shape_idxs[5] = 0
                for idx in [9, 13, 17, 21, 25, 29]:
                    self.in_shape_idxs[idx] = 1
                self.in_shape_idxs[33] = 2

                self.exit_sign_list = [0] * self.block_num
                for idx in self.exit_layers:
                    self.exit_sign_list[idx] = 1

        elif self.model_type == "vgg16_bn":
            self.block_num = 15
            self.in_shape_list = [
                # (3, 224, 224),
                (64, 224, 224),
                (128, 112, 112),
                (256, 56, 56),
                (512, 28, 28),
                (512, 14, 14),
                (512, 7, 7),
            ]

            if self.dataset == "ucf101":
                self.class_num = 101
                self.in_shape_idxs = [None] * self.block_num
                # 特殊情况初始化
                self.in_shape_idxs[1] = 0
                self.in_shape_idxs[3] = 1
                self.in_shape_idxs[5] = 2
                self.in_shape_idxs[7] = 3
                self.in_shape_idxs[9] = 3
                self.in_shape_idxs[11] = 4
                self.in_shape_idxs[13] = 5
                
                self.exit_sign_list = [0] * self.block_num
                for idx in self.exit_layers:
                    self.exit_sign_list[idx] = 1

        self.exit_ths = ths
        self.create_exit_nets()


    def load_init_weights(self, dir):
        for bid in self.exit_layers:
            file = os.path.join(dir, f"{bid}_exit_net_weights.pkl")
            with open(file, 'rb') as fi:
                loaded_data = pickle.load(fi)
              
            model_weight = loaded_data["weights"]
            self.exit_nets[bid].load_state_dict(model_weight)


    def weights_to_cpu(self, dir):
        for bid in self.exit_layers:
            file = os.path.join(dir, f"{bid}_exit_net_weights.pkl")
            with open(file, 'rb') as fi:
                loaded_data = pickle.load(fi)
            
            acc = loaded_data["acc"].to("cpu")
            model_weight = loaded_data["weights"]
            self.exit_nets[bid].load_state_dict(model_weight)
            self.exit_nets[bid].to("cpu")

            save_data = {
                "acc": acc,
                "weights": None
            }

            save_data["weights"] = copy.deepcopy(self.exit_nets[bid].state_dict())

            with open(file, 'wb') as fo:
                pickle.dump(save_data, fo) 
            

    def create_exit_nets(self):
        """ """
        self.exit_nets = dict()
        # self.exit_ths = dict()
        for exit_layer in self.exit_layers:
            # print(self.in_shape_idxs, exit_layer)
            in_shape = self.in_shape_list[ self.in_shape_idxs[exit_layer] ]
            exit_net = Exit_Net(in_shape=in_shape, num_class=self.class_num, model_type=self.model_type)
            exit_net.custom_initialize_weights()
            self.exit_nets[exit_layer] = exit_net

            # self.exit_ths[exit_layer] = self.exit_ths[exit_layer]


    def display_info(self):
        print(f"state: {self.state}")
        print(f"model_type: {self.model_type}")
        print(f"dataset: {self.dataset}")
        print(f"exit_layers: {self.exit_layers}")
        print(f"block_num: {self.block_num}")
        print(f"in_shape_list: {self.in_shape_list}")
        print(f"class_num: {self.class_num}")
        print(f"in_shape_idxs: {self.in_shape_idxs}")
        print(f"exit_sign_list: {self.exit_sign_list}")
        print(f"exit_nets: {self.exit_nets}")

    def save(self, file="mul_exits/mul_exits.pkl"):
        save_data = {
            "state"          : self.state,
            "model_type"     : self.model_type,
            "dataset"        : self.dataset,
            "exit_layers"    : self.exit_layers,
            "block_num"      : self.block_num,
            "in_shape_list"  : self.in_shape_list,
            "class_num"      : self.class_num,
            "in_shape_idxs"  : self.in_shape_idxs,
            "exit_sign_list" : self.exit_sign_list,
            "exit_nets"      : self.exit_nets
        }

        # 保存数据到文件
        with open(file, 'wb') as fo:
            pickle.dump(save_data, fo)

    def load(self, file="mul_exits/mul_exits.pkl"):
        # 从文件加载数据
        with open(file, 'rb') as fi:
            loaded_data = pickle.load(fi)

        self.state          = loaded_data["state"]
        self.model_type     = loaded_data["model_type"]
        self.dataset        = loaded_data["dataset"]
        self.exit_layers    = loaded_data["exit_layers"]
        self.block_num      = loaded_data["block_num"]
        self.in_shape_list  = loaded_data["in_shape_list"]
        self.class_num      = loaded_data["class_num"]
        self.in_shape_idxs  = loaded_data["in_shape_idxs"]
        self.exit_sign_list = loaded_data["exit_sign_list"]
        self.exit_nets      = loaded_data["exit_nets"]


if __name__ == "__main__":
    test_mule = MulExit()
    # test_cache.random_init()

    # test_mule.save()
    # test_mule.load()

    nets_dir = "temp"
    test_mule.weights_to_cpu(nets_dir)
    print("success")

    # test_mule.display_info()
