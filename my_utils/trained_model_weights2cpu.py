import torch
import torch.nn as nn
import torchvision.models as models

from models.zy_AlexNet import AlexNet2
import os

from two_stream_utils import network

if __name__=="__main__":
    server = 407
    device = "cpu"

    if server in [402, 405]:
        model_weights_dir = "/data/wyliang/model_weights"
    elif server == 407:
        model_weights_dir = "/data0/wyliang/model_weights"
    else:
        print(f"error, server {server} not defined")

    model_type_list = ["vgg16_bn"]
    model_type = model_type_list[0]

    num_classes = 101
    
    if model_type == "resnet101":
        loaded_model = network.resnet101(channel=3)
        model_file = os.path.join(model_weights_dir, 'model_best.pth.tar')
        info_dict = torch.load(model_file, map_location=torch.device(device))
        state_dict = info_dict["state_dict"]
        loaded_model.load_state_dict(state_dict)
    elif model_type == "vgg16_bn":
        # 构建模型
        loaded_model = models.vgg16_bn()
        loaded_model.classifier[-1] = nn.Linear(loaded_model.classifier[-1].in_features, num_classes)
        # 加载参数
        model_file = model_weights_dir + "/train_vgg16_bn_ucf101.pth"
        state_dict = torch.load(model_file)
        loaded_model.load_state_dict(state_dict)

    # 转移并保存
    loaded_model.to("cpu")
    torch.save(loaded_model.state_dict(), model_file)


    print(f"{model_file} saved successfully ...")