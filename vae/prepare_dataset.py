import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class prepare_dataset():
    def __init__(self, train = 'dataset/train', valid = 'dataset/valid', low_image_size = 32, high_image_size = 1024, batch_size = 1):
        train_dataset_path = os.path.join(train)
        valid_dataset_path = os.path.join(valid)

        x_transform = transforms.Compose([
            transforms.Resize(low_image_size),
            transforms.CenterCrop(low_image_size),
            transforms.ToTensor()
        ])
        y_transform = transforms.Compose([
            transforms.Resize(high_image_size),
            transforms.CenterCrop(high_image_size),
            transforms.ToTensor()
        ])

        x_dataset = datasets.ImageFolder(train_dataset_path, x_transform)
        self.x_train_loader = DataLoader(x_dataset, batch_size = batch_size)
        x_dataset = datasets.ImageFolder(valid_dataset_path, x_transform)
        self.x_valid_loader = DataLoader(x_dataset, batch_size = 1)


        y_dataset = datasets.ImageFolder(train_dataset_path, y_transform)
        self.y_train_loader = DataLoader(y_dataset, batch_size = batch_size)
        y_dataset = datasets.ImageFolder(valid_dataset_path, y_transform)
        self.y_valid_loader = DataLoader(y_dataset, batch_size = 1)
