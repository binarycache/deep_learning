import numpy
import pandas
import torch
import torchvision
from torchvision.transforms import transforms
import sklearn

class DataLoader():
    """
    python class to load data for cnn.ipynb
    """
    def __init__(self, dataset_name, batch_size=32):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.train_set = None
        self.train_loader = self.load_data()
    
    def load_data(self):
        if self.dataset_name == "fashion_mnist":
            self.train_set = torchvision.datasets.FashionMNIST(
                root='./data/FashionMNIST',
                train=True,
                download=True,
                transform=transforms.Compose([
                            transforms.ToTensor()
                        ]
                    )
                )
            return torch.utils.data.DataLoader(self.train_set, self.batch_size)
        else:
            return None
        
    