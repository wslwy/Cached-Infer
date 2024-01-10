import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets

import os
import copy
import pickle

import data_pre_utils.load_data_v2 as load_data

import logging

# 创建 logger 对象
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 创建文件处理器并设置日志级别
logger_file = "logs/resnet152_detailed_log.log"
file_handler = logging.FileHandler(logger_file)

# 将文件处理器添加到 logger
logger.addHandler(file_handler)


server = 405

if server in [402, 405]:
    save_dir = "/data/wyliang/model_weights"
elif server == 407:
    save_dir = "/data0/wyliang/model_weights"
else:
    print(f"error, server {server} not defined")

def train_model(model_type, model, train_loader, test_loader, num_epochs, device, save_dir=save_dir, target_acc=0.82):
    # 数据集
    dataloaders = {"train": train_loader, "val": test_loader}


    best_train_epoch = -1
    best_train_acc = 0.0 
    best_val_epoch = -1
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
    elif model_type == "resnet152":
        num_classes = 101
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        print("error, model type not defined")

    lr = 0.0005
    momentum = 0.9
    print(model)

    logger.info(f"device        : {device}")
    logger.info(f"lr            : {num_per_class}")
    logger.info(f"num_class     : {num_class}")
    logger.info(f"model: \n {model}")
    # print(model)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # 学习率注意
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # 训练模型
    model.to(device)
    criterion.to(device)
    
    print(f"device: {device}")


    print("training ......")
    logger.info("training ......")
        
    
    stop_flag = False
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
                logger.info(f"epoch: {epoch}, phase: {phase}, Batch: {idx + 1}/{len(dataloaders[phase])}")

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = corrects.double() / len(dataloaders[phase].dataset)

            print(f'Epoch: {epoch + 1}/{num_epochs}, Phase: {phase}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')
            logger.info(f'Epoch: {epoch + 1}/{num_epochs}, Phase: {phase}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

            # 如果是训练阶段并且当前模型性能更好，则保存模型
            if phase == 'train' and epoch_acc > target_acc:
                best_train_epoch = epoch
                best_train_acc = epoch_acc
                try:
                    best_train_model_weights = copy.deepcopy(model.state_dict())
                    torch.save(best_train_model_weights, os.path.join(save_dir, 'train_resnet50_ucf101.pth'))
                    print("Best acc {best_train_acc}, Best model saved!")
                    logger.info("Best acc {best_train_acc}, Best model saved!")
                except Exception as e:
                    print(f"Error saving best model: {e}")
                    logger.info(f"Error saving best model: {e}")
            
                logger.info(f"epoch: {epoch + 1}, phase: {phase}, acc: {epoch_acc}, best_acc: {best_train_acc}\n")
                stop_flag = True
                break

            # 如果是验证阶段并且当前模型性能更好，则保存模型
            if phase == 'val' and epoch_acc > best_val_acc:
                best_val_epoch = epoch
                best_val_acc = epoch_acc
                try:
                    best_val_model_weights = copy.deepcopy(model.state_dict())
                    torch.save(best_val_model_weights, os.path.join(save_dir, 'test_resnet50_ucf101.pth'))
                    print("Best acc {best_val_acc}, Best model saved!")
                    logger.info("Best acc {best_val_acc}, Best model saved!")
                except Exception as e:
                    print(f"Error saving best model: {e}")
                    logger.info(f"Error saving best model: {e}")

                logger.info(f"epoch: {epoch + 1}, phase: {phase}, acc: {epoch_acc}, best_acc: {best_val_acc}\n")

        if stop_flag:
            break
    # # 最终将最好的模型参数保存到文件
    # try:
    #     torch.save(best_val_model_weights, os.path.join(save_dir, 'resnet50_ucf101_01.pth'))
    # except:
    #     pass

    return (best_train_epoch, best_train_acc, best_train_model_weights, best_val_epoch, best_val_acc, best_val_model_weights)


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

    if server in [402, 405]:
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

    # logger 添加注释信息
    logger.info(f"num_per_class : {num_per_class}")
    logger.info(f"num_class     : {num_class}")
    logger.info(f"step          : {step}")
    logger.info(f"len(train_lodeer) : {len(train_loader)}")
    logger.info(f"len(train_dataset): {len(train_loader.dataset)}")
    logger.info(f"len(test_lodeer)  : {len(test_loader)}")
    logger.info(f"len(test_dataset) : {len(test_loader.dataset)}")


    
    model_type_list = ["vgg16_bn", "resnet50", "resnet101", "resnet152"]
    model_type = model_type_list[3]
    num_epochs = 100

    # logger 添加注释信息
    logger.info(f"model_type    : {model_type}")
    logger.info(f"num_epochs    : {num_epochs}")


    if model_type == "vgg16_bn":
        model = models.vgg16_bn(weights='IMAGENET1K_V1')
    elif model_type == "resnet50":
        model = models.resnet50(weights='IMAGENET1K_V1')
    elif model_type == "resnet152":
        model = models.resnet50(weights='IMAGENET1K_V1')
    else:
        print("error, model type not defined")
    # print(model)
        
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    target_acc = 0.82

    return_info = train_model(model_type, model, train_loader, test_loader, num_epochs, device, target_acc)
    best_train_epoch, best_train_acc, best_train_model_weights, best_val_epoch, best_val_acc, best_val_model_weights = return_info

    save_data = {
        "best_train_epoch": best_train_epoch,
        "best_train_acc": best_train_acc,
        "best_val_epoch": best_val_epoch,
        "best_val_acc": best_val_acc
    }

    # logger 添加注释信息
    logger.info(f"best_train_epoch    : {best_train_epoch}")
    logger.info(f"best_train_acc      : {best_train_acc}")
    logger.info(f"best_val_epoch    : {best_val_epoch}")
    logger.info(f"best_val_acc      : {best_val_acc}")

    logger.info(f"best_train_model_weights    : {best_train_model_weights}")
    logger.info(f"best_val_model_weights      : {best_val_model_weights}")

    # 保存数据到文件
    file = "results/resnet50_trained_weights.pkl"
    with open(file, 'wb') as fo:
        pickle.dump(save_data, fo)

    print(save_data)
