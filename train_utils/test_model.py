import torch
import torch.nn as nn
from torchvision import models, transforms, datasets

import os
import copy
import pickle

import data_pre_utils.load_data_v2 as load_data


server = 407

if server == 402:
    save_dir = "/data/wyliang/model_weights"
elif server == 407:
    save_dir = "/data0/wyliang/model_weights"
else:
    print(f"error, server {server} not defined")


def test_model(model, data_loader, device):
    model.to(device)
    model.eval()

    correct = 0

    print(f"{len(data_loader)} batches begin")
    for epc, (data, labels) in enumerate(data_loader):
        data = data.to(device)
        labels = labels.to(device)

        res = model(data)

        _, preds = torch.max(res, 1)
        correct += torch.sum(preds == labels.data)

        print(f"epoch: {epc}, Batch: {epc + 1}/{len(data_loader)}")

    sample_num = len(data_loader.dataset)
    accuracy = float(correct) / sample_num
    print(f"{correct}/{sample_num}, accuracy: {accuracy}")

    return accuracy


def load_model(model_type, num_classes = 101):

    if model_type == "vgg16_bn":
        model = models.vgg16_bn(weights='IMAGENET1K_V1')
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

        # 加载参数
        model_file = "/data0/wyliang/model_weights/vgg16_bn_ucf101.pth"
        # model_file = "/data0/wyliang/model_weights/vgg16_bn_ucf101_01.pth"
        state_dict = torch.load(model_file)
        model.load_state_dict(state_dict)
    elif model_type == "resnet50":
        model = models.resnet50(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        # 加载参数
        model_file = "/data0/wyliang/model_weights/resnet50_ucf101.pth"
        state_dict = torch.load(model_file)
        model.load_state_dict(state_dict)
    else:
        print("error, model type not defined")


    # 将模型参数转 cpu 化保存
    # model.to("cpu")
    # state_dict = copy.deepcopy(model.state_dict())
    # torch.save(state_dict, model_file)
    # model.to("cuda")

    return model


if __name__ =="__main__":
    if server == 402:
        train_dir_list_file = os.path.join("/data/wyliang/datasets/ucf101/ucfTrainTestlist", "trainlist01.txt")
        test_dir_list_file = os.path.join("/data/wyliang/datasets/ucf101/ucfTrainTestlist", "testlist01.txt")
    elif server == 407:
        train_dir_list_file = os.path.join("/data0/wyliang/datasets/ucf101/ucfTrainTestlist", "trainlist01.txt")
        test_dir_list_file = os.path.join("/data0/wyliang/datasets/ucf101/ucfTrainTestlist", "testlist01.txt")


    num_per_class = 10
    num_class = 101
    step = 5

    train_loader = load_data.load_data("ucf101", train_dir_list_file, 64, 256, "train", num_per_class, num_class, step)
    test_loader = load_data.load_data("ucf101", test_dir_list_file, 64, 64, "test", num_per_class, num_class, step)
    print(len(train_loader))
    print(len(train_loader.dataset))
    print(len(test_loader))
    print(len(test_loader.dataset))

    device = "cuda:2"
    model_type_list = ["vgg16_bn", "resnet50"]
    model_type = model_type_list[0]

    model = load_model(model_type)
    accuracy1 = test_model(model, train_loader, device)
    accuracy2 = test_model(model, test_loader, device)
    print(accuracy1, accuracy2)