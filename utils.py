import os
import random
import torch
import shutil
from torchvision.models import *


def save_checkpoint(state, is_best, path='checkpoint', filename='checkpoint.pth.tar'):
    full_path = os.path.join(path, filename)
    torch.save(state, full_path)
    if is_best:
        shutil.copyfile(full_path, os.path.join(path, 'model_best.pth.tar'))
        print("Save best model at %s==" %
              os.path.join(path, 'model_best.pth.tar'))


def model_creator(model_name):
    if model_name == 'resnet18':
        return resnet18(pretrained=True)
    elif model_name == 'resnet34':
        return resnet34(pretrained=True)
    elif model_name == 'resnet50':
        return resnet50(pretrained=True)
    elif model_name == 'resnet101':
        return resnet101(pretrained=True)
    elif model_name == 'resnet152':
        return resnet152(pretrained=True)
    elif model_name == 'resnext50_32x4d':
        return resnext50_32x4d(pretrained=True)
    elif model_name == 'resnext101_32x8d':
        return resnext101_32x8d(pretrained=True)
    elif model_name == 'wide_resnet50_2':
        return wide_resnet50_2(pretrained=True)
    elif model_name == 'wide_resnet101_2':
        return wide_resnet101_2(pretrained=True)
    else:
        return resnet50(pretrained=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
