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
import numpy as np

import yaml


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



class MulUcf101DatasetHelper():
    def __init__(self, image_size: int, mode: str, shuffle: bool, client_num, num_class_matrix, step=10):
        self.datasets = list()

        self.mode = mode
        self.shuffle = shuffle
        self.client_num = client_num
        self.delimiter = delimiter

        self.step = step

        # # 保留意见，是否还用
        self.image_size = image_size

        # Iterate over all image paths
        _, self.class_to_idx = find_classes(ucf101_datasets_root)  # 得到文件夹名到类别编号的字典
        
        # 中间变量，用于控制文件夹数目和文件数
        self.num_class_list = np.array(num_class_matrix)

        folders_list = [ list() for _ in range(self.client_num) ]
        for par_nums, folder in zip(self.num_class_list.T, os.listdir(ucf101_datasets_root)):
            sub_folders = os.listdir( os.path.join(ucf101_datasets_root, folder) )
            if self.shuffle:
                random.shuffle(sub_folders)
            
            idx = 0
            for cnum, num in enumerate(par_nums):
                for sub_folder in sub_folders[idx:idx+num]:
                    folders_list[cnum].append( os.path.join(ucf101_datasets_root, folder, sub_folder) )
                idx += num

        for folders in folders_list:
            client_dataset = Ucf101Dataset(self.image_size, self.mode, folders, self.step)
            self.datasets.append(client_dataset)
        

class Ucf101Dataset(Dataset):
    """Define training/valid dataset loading methods.

    Args:
        image_dir (str): Train/Valid dataset address.
        image_size (int): Image size.
        mode (str): Data set loading method, the training data set is for data enhancement,
            and the verification data set is not for data enhancement.
    """

    def __init__(self, image_size: int, mode: str, folder_list, step=10) -> None:
        super(Ucf101Dataset, self).__init__()
        
        self.mode = mode
        self.delimiter = delimiter

        # # 保留意见，是否还用
        self.image_size = image_size

        # Iterate over all image paths
        _, self.class_to_idx = find_classes(ucf101_datasets_root)  # 得到文件夹名到类别编号的字典
        
        # 根据 视频文件夹选取图片
        self.image_file_paths = []

        # Randomly select images from each class folder
        for img_dir in folder_list:
            image_files = [f for f in os.listdir(img_dir) if f.split(".")[-1].lower() in IMG_EXTENSIONS]  # 检查文件后缀
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
    

def load_data(dataset_type='ucf101', img_size=224, train_batch_size=64, test_batch_size=256, mode="test", shuffle=True, client_num=0, num_class_matrix=None, step=20):
    dataLoaders = list()

    if dataset_type=='ucf101':
        datasets_helper = MulUcf101DatasetHelper(img_size, mode, shuffle, client_num, num_class_matrix, step)

    for dataset in datasets_helper.datasets:
        if mode == "train":
            train_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, num_workers=2)
            data_loader = train_loader
        else:
            test_loader = DataLoader(dataset, batch_size=test_batch_size, shuffle=False, num_workers=2)
            data_loader = test_loader
        
        dataLoaders.append(data_loader)
    
    return dataLoaders



if __name__=="__main__":

    img_size = 224
    mode = "valid"
    shuffle = True
    test_batch_size = 60
    client_num = 4
    step = 5

    base = 3
    num_class_matrix = [
        [base*7] * 10 + [base] * 10 + [base] * 10 + [base] * 10,
        [base] * 10 + [base*7] * 10 + [base] * 10 + [base] * 10,
        [base] * 10 + [base] * 10 + [base*7] * 10 + [base] * 10,
        [base] * 10 + [base] * 10 + [base] * 10 + [base*7] * 10,
    ]
    # print(num_class_matrix)

    # 测试数据集
    # MulData = MulUcf101DatasetHelper(img_size, mode, shuffle, client_num, num_class_matrix, step=5)
    # for data in MulData.datasets:
    #     print(len(data))

    # 测试 Dataloader

    dataLoaders = load_data(test_batch_size=test_batch_size, client_num=4, num_class_matrix=num_class_matrix, step=step)

    for idx, dataloader in enumerate(dataLoaders):
        print(idx, len(dataloader), len(dataloader.dataset))