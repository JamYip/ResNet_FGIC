import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def config(data):
    if data == 'bird':
        train_root = './data/Birds/train'
        test_root = './data/Birds/test'
        train_pd = pd.read_csv("./data/bird_train.txt", sep=" ", header=None,
                               names=['ImageName', 'label'])
        test_pd = pd.read_csv("./data/bird_test.txt", sep=" ", header=None,
                              names=['ImageName', 'label'])
        cls_num = 200

    if data == 'car':
        train_root = './data/Car/train'
        test_root = './data/Car/test'
        train_pd = pd.read_csv("./data/car_train.txt", sep=" ", header=None,
                               names=['ImageName', 'label'])
        test_pd = pd.read_csv("./data/car_test.txt", sep=" ", header=None,
                              names=['ImageName', 'label'])
        cls_num = 196

    if data == 'aircraft':
        train_root = './data/Aircraft/train'
        test_root = './data/Aircraft/test'
        train_pd = pd.read_csv("./data/aircraft_train.txt", sep=" ", header=None,
                               names=['ImageName', 'label'])
        test_pd = pd.read_csv("./data/aircraft_test.txt", sep=" ", header=None,
                              names=['ImageName', 'label'])
        cls_num = 100

    return train_root, test_root, train_pd, test_pd, cls_num


class Dataset(Dataset):
    def __init__(self, root_dir, pd_file, train=False, transform=None):
        self.root_dir = root_dir
        self.pd_file = pd_file
        self.image_names = pd_file['ImageName'].tolist()
        self.labels = pd_file['label'].tolist()

        self.train = train
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        img_path = os.path.join(self.root_dir, self.image_names[item])
        image = self.pil_loader(img_path)
        label = self.labels[item]
        if self.transform:
            image = self.transform(image)
        return image, label

    def pil_loader(self,imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

