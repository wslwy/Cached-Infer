import torch
import torch.nn as nn
import torchvision.models as models

import os
import yaml
import pickle
from my_utils.cache import Cache
import data_pre_utils.load_data_v2 as load_data
# import  my_utils.load_model as load_model

# 设置GPU
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 定义自己的优化器
criterion = torch.nn.CrossEntropyLoss().to(device)

def test_model(model, test_loader, device):
    model.to(device)
    model.eval()

    correct = 0.0
    test_loss = 0.0
    for idx, (data, labels) in enumerate(test_loader):

        data = data.to(device)
        labels = labels.to(device)

        test_out = model(data)

        loss = criterion(test_out, labels)

        test_loss = test_loss + loss.item()
        pred = torch.max(test_out, 1)[1]
        test_correct = (pred == labels).sum()
        correct = correct + test_correct.item()

        print(f"batch {idx} ended ...")

    print('Test_loss: {}, Test correct: {}'.format(test_loss / len(test_loader.dataset), correct / len(test_loader.dataset)))
    return test_loss / len(test_loader.dataset), correct / len(test_loader.dataset)

# model_list前向传播函数
# def forward(model_list, model_type, x, type_list=None):
#     if model_type == "resnet":
#         for type, sub_model in zip(type_list, model_list):
#             # print(type, sub_model)
#             if type == 0:
#                 x = sub_model(x)
#             elif type == 1:
#                 identity = x
#                 x = sub_model(x)
#             elif type == 2:
#                 identity = sub_model(identity)
#             elif type == 3:
#                 x += identity
#                 x = sub_model(x)
#             elif type == 4:
#                 x = torch.flatten(x, 1)
#                 x = sub_model(x)
#     else:
#         for idx, sub_model in enumerate(model_list):
#             if idx == len(model_list) - 1:
#                 x = torch.flatten(x, 1)
#             x = sub_model(x)
        
#     return x

def forward(model_list, model_type, x):
    for idx, sub_model in enumerate(model_list):
        if idx == len(model_list) - 1:
            x = torch.flatten(x, 1)
        x = sub_model(x)
        
    return x


def test_list(model_list, test_loader, model_type, device):
    for sub_model in model_list:
        sub_model.eval()

    correct = 0.0
    test_loss = 0.0


    for idx, (data, labels) in enumerate(test_loader):

        data = data.to(device)
        labels = labels.to(device)

        test_out = forward(model_list, model_type, data)

        loss = criterion(test_out, labels)

        test_loss = test_loss + loss.item()
        pred = torch.max(test_out, 1)[1]
        test_correct = (pred == labels).sum()
        correct = correct + test_correct.item()

        print(f"batch {idx} ended ...")

    print('Test_loss: {}, Test correct: {}'.format(test_loss / len(test_loader.dataset), correct / len(test_loader.dataset)))
    return test_loss / len(test_loader.dataset), correct / len(test_loader.dataset)
    

# def test_list(model_list, test_loader, model_type, device):
#     for sub_model in model_list:
#         sub_model.eval()

#     correct = 0.0
#     test_loss = 0.0

#     # resnet 特有    
#     type_list = [0]
#     type_list += [1, 0, 0, 2, 3] + [1, 0, 0, 3] * 2
#     type_list += [1, 0, 0, 2, 3] + [1, 0, 0, 3] * 3
#     type_list += [1, 0, 0, 2, 3] + [1, 0, 0, 3] * 5
#     type_list += [1, 0, 0, 2, 3] + [1, 0, 0, 3] * 2
#     type_list += [0, 4]

#     for idx, (data, labels) in enumerate(test_loader):

#         data = data.to(device)
#         labels = labels.to(device)

#         test_out = forward(model_list, model_type, data, type_list=type_list)

#         loss = criterion(test_out, labels)

#         test_loss = test_loss + loss.item()
#         pred = torch.max(test_out, 1)[1]
#         test_correct = (pred == labels).sum()
#         correct = correct + test_correct.item()

#         print(f"batch {idx} ended ...")

#     print('Test_loss: {}, Test correct: {}'.format(test_loss / len(test_loader.dataset), correct / len(test_loader.dataset)))
#     return test_loss / len(test_loader.dataset), correct / len(test_loader.dataset)

if __name__ == "__main__":
    # 加载模型架构并加载权重
    device = "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_type_list = ["imagenet1k", "imagenet-100", "ucf101"]
    model_type_list = ["vgg16_bn", "resnet50", "resnet101"]
    
    dataset_type = dataset_type_list[2]
    model_type = model_type_list[0]
    
    # 读取配置文件
    with open('config.yml', 'r') as config_file:
        config = yaml.safe_load(config_file)
        server = config["server"]
        train_dir_list_file = os.path.join(config["datasets"][server]["image_list_dir"], "trainlist01.txt")
        test_dir_list_file = os.path.join(config["datasets"][server]["image_list_dir"], "testlist01.txt")

    # 加载数据集和模型
    train_loader = load_data.load_data("ucf101", train_dir_list_file, 32, 256, "train", 300, 101, 20)
    # test_loader = load_data.load_data(dataset_type, img_dir_list_file, 64, 256, "test", 10, 101, 20)
    # loaded_model = load_model.load_model(device=device, model_type=model_type, dataset_type=dataset_type)
    data_loader = train_loader

    print(len(data_loader))
    print(len(data_loader.dataset))

    # 测试加载模型
    # print(loaded_model)
    # print(loaded_model.children())
    # for idx, layer in enumerate(loaded_model.children()):
    #     print(idx, layer)
    loaded_model = models.vgg16_bn(weights='IMAGENET1K_V1')
    if model_type == "vgg16_bn":
        num_classes = 101
        loaded_model.classifier[-1] = nn.Linear(loaded_model.classifier[-1].in_features, num_classes)

    # with open("results/vgg16_bn_trained_weights.pkl", "rb") as file:
    #     ckp = pickle.load(file)
    # # ckp = torch.load("results/vgg16_bn_trained_weights.pkl")
    # state_dict = ckp["params"]

    state_dict = torch.load("/data0/wyliang/model_weights/vgg16_bn_ucf101.pth")
    loaded_model.load_state_dict(state_dict)
    
    print("begin test model ...")
    test_model(loaded_model, data_loader, device)
    print("test model end ...")

    # 模型划分
    # sub_models = load_model.model_partition(loaded_model, model_type=model_type)

    # 测试划分的模型
    # print(sub_models)
    # for idx, sub_model in enumerate(sub_models):
    #     print(idx, sub_model)
    # print("begin test model list ...")
    # test_list(sub_models, test_loader, model_type="resnet", device=device)
    # print("model list test end ...")

    # 缓存
    # cache = Cache()
    # cache.init_device(device)
    # # test_generate_cache(sub_models, test_loader, device="cpu", cache=cache)

    # file = "./cache/cache.pkl"
    # # cache.save(file)

    # loaded_cache = Cache()
    # loaded_cache.load(file)

    # sum1 = 0
    # sum2 = 0
    # for label in range(100):
    #     print(label, loaded_cache.up_freq_table[0][label], loaded_cache.freq_table[label])
    #     sum1 += loaded_cache.up_freq_table[0][label]
    #     sum2 += loaded_cache.freq_table[label]
    # print(sum1, sum2)
