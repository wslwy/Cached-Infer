import torch
import torch.nn as nn
import torch.optim as optim
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

def train_model(model_type, model, train_loader, test_loader, num_epochs, save_dir=save_dir):
    # 数据集
    dataloaders = {"train": train_loader, "val": test_loader}


    best_epoch = -1
    best_val_acc = 0.0 
    # 模型
    if model_type == "vgg16_bn":
        # # 使用预训练的VGG16_bn模型
        # model = models.vgg16_bn(pretrained=True)

        # 固定前面的层参数，只微调后面的分类层(都训练，微调)
        # for param in model.features.parameters():
        #     param.requires_grad = False

        # 修改分类层的输出类别数为UCF101数据集的类别数（例如101类）
        num_classes = 101
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

        # 加载参数
        with open("results/vgg16_bn_trained_weights.pkl", "rb") as file:
            ckp = pickle.load(file)
        # ckp = torch.load("results/vgg16_bn_trained_weights.pkl")
        print(ckp["acc"])
        model.load_state_dict(ckp["params"])
        best_val_acc = ckp["acc"]
    elif model_type == "resnet50":
        num_classes = 101
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        print("error, model type not defined")

    print(model)
    # print(model)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # 学习率注意
    optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)

    # 训练模型
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion.to(device)
    
    print(f"device: {device}")


    print("training ......")

    log_file = os.path.join("/home/wyliang/Neurosurgeon/logs", 'train_log.txt')
    with open(log_file, 'w') as log_file:
        
        # 这个部分也许可以重写拆分
        for epoch in range(num_epochs):
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                corrects = 0

                for idx, (inputs, labels) in enumerate(dataloaders[phase]):
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    corrects += torch.sum(preds == labels.data)

                    
                    print(f"epoch: {epoch}, phase: {phase}, Batch: {idx + 1}/{len(dataloaders[phase])}")

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = corrects.double() / len(dataloaders[phase].dataset)

                print(f'Epoch: {epoch + 1}/{num_epochs}, Phase: {phase}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

                # 如果是验证阶段并且当前模型性能更好，则保存模型
                if phase == 'val' and epoch_acc > best_val_acc:
                    best_epoch = epoch
                    best_val_acc = epoch_acc
                    try:
                        best_model_weights = copy.deepcopy(model.state_dict())
                        torch.save(best_model_weights, os.path.join(save_dir, 'resnet50_ucf101.pth'))
                        print("Best acc {best_val_acc}, Best model saved!")
                    except Exception as e:
                        print(f"Error saving best model: {e}")

                log_file.write(f"epoch: {epoch + 1}, phase: {phase}, acc: {epoch_acc}, best_acc: {best_val_acc}\n")

    # # 最终将最好的模型参数保存到文件
    # try:
    #     torch.save(best_model_weights, os.path.join(save_dir, 'resnet50_ucf101_01.pth'))
    # except:
    #     pass

    return best_epoch, best_val_acc, best_model_weights


if __name__ == "__main__":
    # 定义数据预处理和加载数据集
    # data_transforms = {
    #     'train': transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     ]),
    #     'val': transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     ]),
    # }
    if server == 402:
        train_dir_list_file = os.path.join("/data/wyliang/datasets/ucf101/ucfTrainTestlist", "trainlist01.txt")
        test_dir_list_file = os.path.join("/data/wyliang/datasets/ucf101/ucfTrainTestlist", "testlist01.txt")
    elif server == 407:
        train_dir_list_file = os.path.join("/data0/wyliang/datasets/ucf101/ucfTrainTestlist", "trainlist01.txt")
        test_dir_list_file = os.path.join("/data0/wyliang/datasets/ucf101/ucfTrainTestlist", "testlist01.txt")

    num_per_class = 300
    num_class = 101
    step = 20
    train_loader = load_data.load_data("ucf101", train_dir_list_file, 32, 256, "train", num_per_class, num_class, step)
    test_loader = load_data.load_data("ucf101", test_dir_list_file, 64, 32, "test", num_per_class, num_class, step)


    model_type = "resnet50"
    num_epochs = 100


    if model_type == "vgg16_bn":
        model = models.vgg16_bn(weights='IMAGENET1K_V1')
    elif model_type == "resnet50":
        model = models.resnet50(weights='IMAGENET1K_V1')
    else:
        print("error, model type not defined")
    # print(model)


    best_epoch, best_val_acc, best_model_params = train_model(model_type, model, train_loader, test_loader, num_epochs)

    save_data = {
        "epoch": best_epoch,
        "acc": best_val_acc,
        "params": best_model_params
    }

    # 保存数据到文件
    file = "results/resnet50_trained_weights_01.pkl"
    print(best_epoch, best_val_acc)
    with open(file, 'wb') as fo:
        pickle.dump(save_data, fo)

    print(best_epoch, best_val_acc)
