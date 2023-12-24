import torch
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.datasets.folder import find_classes
from torch.utils.data import Dataset, Subset, DataLoader

import os
import cv2
import sys
from PIL import Image
from glob import glob
from data_pre_utils import imgproc
import random
import re

import yaml


# 设置随机数种子
seed_value = 2024
random.seed(seed_value)


# 读取配置文件
with open('config.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)

    server = config["server"]
    # print(config)
    if server == 407:
        cifar_datasets_root = config["datasets"][407]["cifar_datasets_root"]
        imagenet_1k_datasets_root = config["datasets"][407]["cifar_datasets_root"]
        imagenet_100_datasets_train_root = config["datasets"][407]["cifar_datasets_root"]
        imagenet_100_datasets_test_root = config["datasets"][407]["cifar_datasets_root"]
        ucf101_datasets_root = config["datasets"][407]["ucf101_datasets_root"]
    elif server == 402:
        ucf101_datasets_root = config["datasets"][402]["ucf101_datasets_root"]

default_image_size = config["default_image_size"]

# Image formats supported by the image processing library
IMG_EXTENSIONS = ("jpg", "jpeg", "png", "ppm", "bmp", "pgm", "tif", "tiff", "webp")

# The delimiter is not the same between different platforms
if sys.platform == "win32":
    delimiter = "\\"
else:
    delimiter = "/"

class ImageDataset(Dataset):
    """Define training/valid dataset loading methods.

    Args:
        image_dir (str): Train/Valid dataset address.
        image_size (int): Image size.
        mode (str): Data set loading method, the training data set is for data enhancement,
            and the verification data set is not for data enhancement.
    """

    def __init__(self, image_dir: str, image_size: int, mode: str, num_per_class: int, num_class=100) -> None:
        super(ImageDataset, self).__init__()
        # Iterate over all image paths
        # self.image_file_paths = glob(f"{image_dir}/*/*")    # 得到所有图片文件路径的list
        # Form image class label pairs by the folder where the image is located
        _, self.class_to_idx = find_classes(image_dir)  # 得到文件夹名到类别编号的字典
        self.image_dir = image_dir
        self.image_size = image_size
        self.mode = mode
        self.num_class = num_class
        self.num_per_class = num_per_class
        self.delimiter = delimiter

        # 按照每个类多少图片进行图片路径抽取
        # 获取原始数据集中的类别文件夹列表
        self.class_folders = os.listdir(image_dir)[:self.num_class]

        # Initialize a list to store the image file paths
        self.image_file_paths = []

        # Randomly select images from each class folder
        for class_folder in self.class_folders:
            class_dir = os.path.join(image_dir, class_folder)   # 获取文件夹目录
            image_files = [f for f in os.listdir(class_dir) if f.split(".")[-1].lower() in IMG_EXTENSIONS]  # 检查文件后缀
            random.shuffle(image_files) #打乱
            selected_images = image_files[:min(self.num_per_class, len(image_files))]
            self.image_file_paths.extend([os.path.join(class_dir, image) for image in selected_images])

        if self.mode == "train":
            # Use PyTorch's own data enhancement to enlarge and enhance data
            self.pre_transform = transforms.Compose([
                transforms.RandomResizedCrop(self.image_size),
                TrivialAugmentWide(),
                transforms.RandomRotation([0, 270]),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
            ])
        elif self.mode == "valid" or self.mode == "test":
            # Use PyTorch's own data enhancement to enlarge and enhance data
            self.pre_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop([self.image_size, self.image_size]),
            ])
        else:
            raise "Unsupported data read type. Please use `Train` or `Valid` or `Test`"

        self.post_transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, batch_index: int) -> [torch.Tensor, int]:
        image_dir, image_name = self.image_file_paths[batch_index].split(self.delimiter)[-2:]
        # Read a batch of image data
        if image_name.split(".")[-1].lower() in IMG_EXTENSIONS:
            image = cv2.imread(self.image_file_paths[batch_index])
            label = self.class_to_idx[image_dir]
        else:
            print(image_name.split(".")[-1].lower())
            raise ValueError(f"Unsupported image extensions, Only support `{IMG_EXTENSIONS}`, "
                             "please check the image file extensions.")

        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # OpenCV convert PIL
        image = Image.fromarray(image)

        # Data preprocess
        image = self.pre_transform(image)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        tensor = imgproc.image_to_tensor(image, False, False)

        # Data postprocess
        tensor = self.post_transform(tensor)

        # return {"image": tensor, "label": label}
        return tensor, label

    def __len__(self) -> int:
        return len(self.image_file_paths) 

def get_cifar_100_dataset():
        # # 训练集的转换
    # train_transform = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),  # 随机裁剪到32x32大小
    #     transforms.RandomHorizontalFlip(),  # 随机水平翻转
    #     transforms.ToTensor(),  # 转换为Tensor
    #     transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])  # 标准化
    # ])

    # # 测试集的转换
    # test_transform = transforms.Compose([
    #     transforms.ToTensor(),  # 转换为Tensor
    #     transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])  # 标准化
    # ])

    #### 另一种 transform 实现思路
    # 基础 tansform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    # additional_transform
    additional_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪到32x32大小
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
    ])

    # 加载完整的 CIFAR-100 数据集 
    full_dataset = tv.datasets.CIFAR100(root=cifar_datasets_root, train=True, transform=transform, download=False)
    # print(len(full_dataset))

    # 划分训练集和测试集的索引
    test_ratio = 0.2
    k = int(len(full_dataset) * (1-test_ratio))
    train_indices = range(0, k)  # 前80%个样本用于训练
    test_indices = range(k, len(full_dataset))  # 后20%个样本用于测试

    # 创建训练集和测试集的子集
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # 训练集添加额外的 transform
    train_dataset.dataset.transform = transforms.Compose([
        train_dataset.dataset.transform,
        additional_transform
    ])

    return train_dataset, test_dataset

class Ucf101Dataset(Dataset):
    """Define training/valid dataset loading methods.

    Args:
        image_dir (str): Train/Valid dataset address.
        image_size (int): Image size.
        mode (str): Data set loading method, the training data set is for data enhancement,
            and the verification data set is not for data enhancement.
    """

    def __init__(self, image_dir_list_file: str, image_size: int, mode: str, shuffle: bool, num_per_class=2, num_class=2, step=10) -> None:
        super(Ucf101Dataset, self).__init__()
        
        self.image_dir_list_file = image_dir_list_file
        self.shuffle = shuffle
        self.mode = mode
        self.delimiter = delimiter

        # # 保留意见，是否还用
        self.num_class = num_class
        self.num_per_class = num_per_class
        # 按照每个类多少图片进行图片路径抽取
        # 获取原始数据集中的类别文件夹列表
        # self.class_folders = os.listdir(ucf101_datasets_root)[:self.num_class]
        # # 图片不一定需要裁剪
        self.image_size = image_size

        # Iterate over all image paths
        _, self.class_to_idx = find_classes(ucf101_datasets_root)  # 得到文件夹名到类别编号的字典
        
        # 中间变量，用于控制文件夹数目和文件数
        class_dict = dict()
        select_lines = list()
        class_num = 0

        with open(self.image_dir_list_file, 'r') as file:
            # 逐行读取文件内容
            file_lines = file.readlines() 

        if self.shuffle:
            random.shuffle(file_lines)

        for file in file_lines:
            dir = file.split("/")[0]
            if dir in class_dict:
                if class_dict[dir] < self.num_per_class:
                    select_lines.append(file)
                    class_dict[dir] += 1
                else:
                    pass
            else:
                if class_num < self.num_class:
                    select_lines.append(file)
                    class_num += 1
                    class_dict[dir] = 1
                else:
                    # impossible
                    pass


        # print(file_lines[:20])
        # print(select_lines)

        # 按照每个类多少图片进行图片路径抽取
        # 获取原始数据集中的类别文件夹列表
        folder_list = []

        # 是否需要 train 和 test 分开处理
        for line in select_lines:
            folder = line.split()[0].split(".")[0]
            folder_list.append(folder)

        # 根据 视频文件夹选取图片
        self.image_file_paths = []

        # Randomly select images from each class folder
        for folder in folder_list:
            img_dir = os.path.join(ucf101_datasets_root, folder)   # 获取文件夹目录
            image_files = [f for f in os.listdir(img_dir) if f.split(".")[-1].lower() in IMG_EXTENSIONS]  # 检查文件后缀
            # random.shuffle(image_files) #打乱
            # selected_images = image_files[:min(self.num_per_class, len(image_files))]
            selected_images = sorted(image_files, key=lambda x: int(re.findall(r'\d+', x)[-1]))  # 按照字典序排列
            selected_images = selected_images[::step]   # 压缩数据集大小，根据一定步长取样
            self.image_file_paths.extend([os.path.join(img_dir, image) for image in selected_images])

        # test部分
        # test = self.image_file_paths[:50]
        # print(len(self.image_file_paths))
        # for test_img in test:
        #     print(test_img)

        if self.mode == "train":
            # Use PyTorch's own data enhancement to enlarge and enhance data
            self.pre_transform = transforms.Compose([
                transforms.RandomResizedCrop(self.image_size),
                # TrivialAugmentWide(),
                transforms.RandomRotation([0, 270]),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
            ])
        elif self.mode == "valid" or self.mode == "test":
            # Use PyTorch's own data enhancement to enlarge and enhance data
            self.pre_transform = transforms.Compose([
                # 考虑到 ucf101 分辨率 320 X 240
                transforms.Resize(240),
                transforms.CenterCrop([self.image_size, self.image_size]),
            ])
        else:
            raise "Unsupported data read type. Please use `train` or `valid` or `test`"

        self.post_transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, batch_index: int) -> [torch.Tensor, int]:
        image_dir, _, image_name = self.image_file_paths[batch_index].split(self.delimiter)[-3:]
        # Read a batch of image data
        if image_name.split(".")[-1].lower() in IMG_EXTENSIONS:
            image = cv2.imread(self.image_file_paths[batch_index])
            label = self.class_to_idx[image_dir]
        else:
            print(image_name.split(".")[-1].lower())
            raise ValueError(f"Unsupported image extensions, Only support `{IMG_EXTENSIONS}`, "
                             "please check the image file extensions.")

        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # OpenCV convert PIL
        image = Image.fromarray(image)

        # Data preprocess
        image = self.pre_transform(image)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        tensor = imgproc.image_to_tensor(image, False, False)

        # Data postprocess
        tensor = self.post_transform(tensor)

        # return {"image": tensor, "label": label}
        return tensor, label

    def __len__(self) -> int:
        return len(self.image_file_paths)
   

def load_data(dataset='ucf101', img_dir_list_file=None, train_batch_size=64, test_batch_size=256, mode="test", num_per_class=300, num_class=100, step=20):
    if dataset == "cifar-100":
        train_dataset, test_dataset = get_cifar_100_dataset()
    elif dataset == 'imagenet-1k':
        if mode == "train":
            train_dataset = ImageDataset(image_dir=imagenet_1k_datasets_root, image_size=default_image_size, mode=mode)
        else:
            test_dataset =ImageDataset(image_dir=imagenet_1k_datasets_root, image_size=default_image_size, mode=mode)
    elif dataset == 'imagenet-100':
        if mode == "train":
            train_dataset = ImageDataset(image_dir=imagenet_100_datasets_train_root, image_size=default_image_size, mode="test", num_per_class=num_per_class, num_class=num_class)
        else:
            test_dataset = ImageDataset(image_dir=imagenet_100_datasets_test_root, image_size=default_image_size, mode=mode, num_per_class=num_per_class, num_class=num_class)
    elif dataset == 'ucf101':
        if mode == "train":
            train_dataset = Ucf101Dataset(image_dir_list_file=img_dir_list_file, image_size=default_image_size, mode=mode, shuffle=True, num_per_class=num_per_class, num_class=num_class, step=step)
        else:
            test_dataset = Ucf101Dataset(image_dir_list_file=img_dir_list_file, image_size=default_image_size, mode=mode, shuffle=True, num_per_class=num_per_class, num_class=num_class, step=step)
    else:
        print("error, no dataset matched")

    # 创建训练集和测试集的数据加载器 (num_workers 看看是否需要设为1),根据需要定义合适的dataloader, shuffle是否为True
    if mode == "train":
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2)
        data_loader = train_loader
    else:
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2)
        data_loader = test_loader
    return data_loader
    

if __name__ == "__main__":
    # train_loader, test_loader = load_data()

    # print(len(train_loader.dataset))
    # print(len(test_loader.dataset))


    # # 验证 dataset 类定义
    # img_dir = "/data0/zxie/zxie/imagenet1000/val"
    
    # server = 407
    # # server
    # if server == 407:
    #     dir_list_dir = "/data0/wyliang/datasets/ucf101/ucfTrainTestlist"
    # elif server == 402:
    #     dir_list_dir = "/data/wyliang/datasets/ucf101/ucfTrainTestlist"
    # # img_dir_list_file = os.path.join(dir_list_dir, "trainlist01.txt")
    # img_dir_list_file = os.path.join(dir_list_dir, "testlist01.txt")
    # img_size = 224
    # mode = "valid"
    # shuffle = True

    # dataset = Ucf101Dataset(img_dir_list_file, img_size, mode, shuffle)
    # print("image_file_paths:", type(dataset.image_file_paths), len(dataset.image_file_paths))
    # img, label = dataset.__getitem__(0)
    # print(img, img.shape, label)
    # print("image_file_paths:", dataset.image_file_paths)
    # print(dataset.class_to_idx)
    # print(len(dataset))

    # dataset = ImageDataset(img_dir, img_size, mode)
    # print("image_file_paths:", type(dataset.image_file_paths), len(dataset.image_file_paths))
    # img, label = dataset.__getitem__(0)
    # print(img, img.shape, label)
    # print("image_file_paths:", dataset.image_file_paths)
    # print(dataset.class_to_idx)
    # print(dataset)

    


    # test_loader = load_data("ucf101", img_dir_list_file, 64, 256, "test", 1, 1, 5)
    # print(len(test_loader))
    # print(len(test_loader.dataset))
    # img, label = test_loader.dataset.__getitem__(0)
    # print(img, img.shape, label)

    # # # 将 DataLoader 转换为迭代器
    # # data_iterator = iter(test_loader)

    # # # 获取一个 batch 的数据
    # # images, labels = next(data_iterator)
    # # # 这里的 batch 包含了一个 batch 的数据
    # # print(images.shape)
    # # print("Batch Data:", labels)


    # 加载测试数据
    if server == 402:
        train_dir_list_file = os.path.join("/data/wyliang/datasets/ucf101/ucfTrainTestlist", "trainlist01.txt")
        test_dir_list_file = os.path.join("/data/wyliang/datasets/ucf101/ucfTrainTestlist", "testlist01.txt")
    elif server == 407:
        train_dir_list_file = os.path.join("/data0/wyliang/datasets/ucf101/ucfTrainTestlist", "trainlist01.txt")
        test_dir_list_file = os.path.join("/data0/wyliang/datasets/ucf101/ucfTrainTestlist", "testlist01.txt")


    class_num = 50  # 101
    num_per_class = 10   #  10000   # 足够大
    batch_size = 60

    step = 5
    train_loader = load_data("ucf101", train_dir_list_file, 64, 256, "train", 15, class_num, 5)
    test_loader = load_data("ucf101", test_dir_list_file, 64, batch_size, "test", num_per_class, class_num, step)
    print(len(train_loader))
    print(len(train_loader.dataset))
    print(len(test_loader))
    print(len(test_loader.dataset))