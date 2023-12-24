import torch
import torchvision.models as models
from torchsummary import summary


if __name__=="__main__":
    model_types = ["vgg16_bn", "resnet50", "resnet101", "googlenet", "mobilenet_v3_large"]
    idx = 4
    model_type = model_types[idx]

    device = "cpu"

    if model_type == "vgg16_bn":
        model = models.vgg16_bn(weights='IMAGENET1K_V1')
    elif model_type == "resnet50":
        model = models.resnet50(weights='IMAGENET1K_V1')
    elif model_type == "resnet101":
        model = models.resnet101()
    elif model_type == "googlenet":
        model = models.googlenet()
    elif model_type == "mobilenet_v3_large":
        model = models.mobilenet_v3_large()
    model.to(device)

    summary(model, (3, 224, 224))