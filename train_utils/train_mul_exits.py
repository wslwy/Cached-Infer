import torch
import torch.nn as nn
import torch.optim as optim

import os
import pickle
import yaml
import copy
import numpy as np

import data_pre_utils.load_data_v2 as load_data
import  my_utils.load_model as load_model
from my_utils.mul_exit import MulExit


def forward(model_list, x, y, tar_dir, idx2class, file_cnts, layer_signs):
    """
    根据切分好的模型进行中间结果生成
    """
    for idx, sub_model in enumerate(model_list):
        if idx == len(model_list) - 1:
            x = torch.flatten(x, 1)
        x = sub_model(x)

        if layer_signs[idx]:
            # file_path = os.path.join(tar_dir, str(idx), idx2class[y], str(file_cnts[y])+".pth")
            # 保存文件
            # torch.save(x, file_path)
            # file_cnts[y] += 1
            print(idx, x.shape)
            
    return x

def train_mul_exits(model_list, train_loader, test_loader, device, mul_exits, num_epochs=50, nets_dir=None):
    # 数据集
    dataloaders = {"train": train_loader, "val": test_loader}

    # 定义损失函数和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    for sub_model in model_list:
        sub_model.eval()
        sub_model.to(device)
        # 冻结主干网络参数
        for param in sub_model.parameters():
            param.requires_grad = False

    optimizers = dict()
    for key, exit_net in mul_exits.exit_nets.items():
        exit_net.to(device)
        optimizers[key] = optim.Adam(exit_net.parameters(), lr=0.001)
        # optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)


    best_epochs = dict()
    best_val_accs = dict()
    for bid in mul_exits.exit_layers:
        best_epochs[bid] = -1
        if nets_dir:
            file = os.path.join(nets_dir, f"{bid}_exit_net_weights.pkl")
            with open(file, 'rb') as fi:
                loaded_data = pickle.load(fi)
            
            best_val_accs[bid] = loaded_data["acc"]
        else:
            best_val_accs[bid] = 0.0

    print("training ......")

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                for exit_net in mul_exits.exit_nets.values():
                    exit_net.train()
            else:
                for exit_net in mul_exits.exit_nets.values():
                    exit_net.eval()

            losses = dict()
            corrects = dict()
            for key in mul_exits.exit_nets.keys():
                losses[key] = 0.0
                corrects[key] = 0

            for batch, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs, labels = inputs.to(device), labels.to(device)

                run_losses = dict()
                mid_outs = dict()

                # zero_grad
                if phase == "train":
                    for bid in mul_exits.exit_layers:
                        optimizers[bid].zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    x = inputs
                    for bid, sub_model in enumerate(model_list):
                        if bid == len(model_list) - 1:
                            x = torch.flatten(x, 1)
                        x = sub_model(x)

                        if mul_exits.exit_sign_list[bid] == 1:
                            # print(bid, x.shape, mul_exits.exit_nets[bid])
                            mid_outs[bid] = mul_exits.exit_nets[bid](x)
                            run_losses[bid] = criterion(mid_outs[bid], labels)

                    # backward & step
                    if phase == "train":
                        for bid in mul_exits.exit_layers:
                            run_losses[bid].backward()
                            optimizers[bid].step()
                
                for bid in mul_exits.exit_layers:
                    losses[bid] += run_losses[bid].item() * inputs.size(0)
                    _, preds = torch.max(mid_outs[bid], 1)
                    corrects[bid] += torch.sum(preds == labels.data)
                
                print(f"epoch: {epoch}, phase: {phase}, Batch: {batch + 1}/{len(dataloaders[phase])}")

            print(f'Epoch: {epoch + 1}/{num_epochs}, Phase: {phase}')
            for bid in mul_exits.exit_layers:
                losses[bid] /= len(dataloaders[phase].dataset)
                corrects[bid] = corrects[bid].double() / len(dataloaders[phase].dataset)
                print(f"bid: {bid}, Loss: {losses[bid]:.4f}, Acc: {corrects[bid]:.4f}")
            
                # 如果是验证阶段并且当前模型性能更好，则保存模型
                if phase == 'val' and corrects[bid]  > best_val_accs[bid]:
                    best_epochs[bid] = epoch
                    best_val_accs[bid] = corrects[bid]
                    try:
                        best_model_weights = copy.deepcopy(mul_exits.exit_nets[bid].state_dict())
                        file_path = os.path.join("mule_vgg16_bn", f"{bid}_exit_net_weights.pkl")
                        # 保存
                        save_data = {
                            "acc" : corrects[bid],
                            "weights": best_model_weights
                        }
                        with open(file_path, 'wb') as fo:
                            pickle.dump(save_data, fo)                    
                        print(f"bid: {bid}, Best acc {best_val_accs[bid]}, Best model saved!")
                    except Exception as e:
                        print(f"Error saving best model: {e}")

    return


if __name__ == "__main__":
    # 读取配置文件
    with open('config.yml', 'r') as config_file:
        config = yaml.safe_load(config_file)
        server = config["server"]
        img_dir_list_file = os.path.join(config["datasets"][server]["image_list_dir"], "trainlist01.txt")


    batch_size = 60
    device = "cpu"
    dataset_type_list = ["imagenet1k", "imagenet-100", "ucf101"]
    model_type_list = ["vgg16_bn", "resnet50", "resnet101"]
    
    dataset_type = dataset_type_list[2]
    model_type = model_type_list[0]

    num_epochs = 100
    nets_dir = "mule_vgg16_bn"

    # 加载模型
    loaded_model = load_model.load_model(device=device, model_type=model_type, dataset_type=dataset_type)
    # 模型划分
    sub_models = load_model.model_partition(loaded_model, model_type)
    print(len(sub_models))

    class_num = 101
    num_per_class = 200   # 足够大

    # 加载测试数据
    if server == 402:
        train_dir_list_file = os.path.join("/data/wyliang/datasets/ucf101/ucfTrainTestlist", "trainlist01.txt")
        test_dir_list_file = os.path.join("/data/wyliang/datasets/ucf101/ucfTrainTestlist", "testlist01.txt")
    elif server == 407:
        train_dir_list_file = os.path.join("/data0/wyliang/datasets/ucf101/ucfTrainTestlist", "trainlist01.txt")
        test_dir_list_file = os.path.join("/data0/wyliang/datasets/ucf101/ucfTrainTestlist", "testlist01.txt")

    train_loader = load_data.load_data("ucf101", train_dir_list_file, 64, 256, "train", 25, class_num, 5)
    test_loader = load_data.load_data("ucf101", test_dir_list_file, 64, 64, "test", 15, class_num, 20)
    print(len(train_loader))
    print(len(train_loader.dataset))
    print(len(test_loader))
    print(len(test_loader.dataset))

    # 生成多出口类
    mul_exits = MulExit(state="global", model_type=model_type, dataset=dataset_type, exit_layers = [1,3,5,7,9,11,13])
    mul_exits.load_init_weights(nets_dir) # 加载参数
    # mul_exits.display_info()

    # 生成保存中间结果
    

    print("training begin")
    train_mul_exits(sub_models, train_loader, test_loader, device, mul_exits, num_epochs=num_epochs, nets_dir=nets_dir)
    print("training finished")


    # # 保存
    # save_data = {
    #     "acc" : accuracy
    # }

    # with open("results/acc_record.pkl", 'wb') as fo:
    #     pickle.dump(save_data, fo)


    # print(save_data)