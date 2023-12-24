import torch
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.datasets.folder import find_classes
from torch.utils.data import Dataset, Subset, DataLoader

import cv2
import sys
from PIL import Image
from glob import glob
import imgproc

cifar_datasets_root = '/data/wyliang/datasets/CIFAR-100'
imagenet_1k_datasets_root = "/data0/zxie/zxie/imagenet1000/val"
imagenet_100_datasets_train_root = "/data0/zxie/zxie/IMAGE100/train"
imagenet_100_datasets_test_root = "/data0/zxie/zxie/IMAGE100/test"

# default_image_size = 224
default_image_size = 256

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

    def __init__(self, image_dir: str, image_size: int, mode: str) -> None:
        super(ImageDataset, self).__init__()
        # Iterate over all image paths
        self.image_file_paths = glob(f"{image_dir}/*/*")    # 得到所有图片文件路径的list
        # Form image class label pairs by the folder where the image is located
        _, self.class_to_idx = find_classes(image_dir)  # 得到文件夹名到类别编号的字典
        self.image_size = image_size
        self.mode = mode
        self.delimiter = delimiter

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
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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

def load_data(dataset='imagenet-100', train_batch_size=64, test_batch_size=256, mode="test"):
    if dataset == "cifar-100":
        train_dataset, test_dataset = get_cifar_100_dataset()
    elif dataset == 'imagenet-1k':
        if mode == "train":
            train_dataset = ImageDataset(image_dir=imagenet_1k_datasets_root, image_size=default_image_size, mode=mode)
        else:
            test_dataset =ImageDataset(image_dir=imagenet_1k_datasets_root, image_size=default_image_size, mode=mode)
    elif dataset == 'imagenet-100':
        if mode == "train":
            train_dataset = ImageDataset(image_dir=imagenet_100_datasets_train_root, image_size=default_image_size, mode="test")
        else:
            test_dataset =ImageDataset(image_dir=imagenet_100_datasets_test_root, image_size=default_image_size, mode=mode)


    # 创建训练集和测试集的数据加载器 (num_workers 看看是否需要设为1),根据需要定义合适的dataloader
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


    # 验证 imagenet dataset 类定义
    # img_dir = "/data0/zxie/zxie/imagenet1000/val"
    # img_size = 256
    # mode = "Valid"

    # dataset = ImageDataset(img_dir, img_size, mode)
    # print("image_file_paths:", type(dataset.image_file_paths), len(dataset.image_file_paths))
    # img, label = dataset.__getitem__(0)
    # print(img, img.shape, label)
    # print("image_file_paths:", dataset.image_file_paths)
    # print(dataset.class_to_idx)
    # print(dataset)
    test_loader = load_data("imagenet-100", 64, 256, "test")
    print(len(test_loader))
    print(len(test_loader.dataset))
    img, label = test_loader.dataset.__getitem__(0)
    print(img, img.shape, label)