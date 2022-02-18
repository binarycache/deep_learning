import numpy
import pandas
import torch
import torchvision
from torchvision.transforms import transforms
import sklearn


class DataLoader:
    """
    python class to load data.
    """

    def __init__(self, dataset_name, normalize=False, batch_size=32, num_workers=0):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_set = self.load_data()
        self.train_loader = self.loader(normalize)

    def load_data(self):
        return torchvision.datasets.FashionMNIST(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        )

    def __calculate_stats(self):
        """
        Returns the standard deviation and variance of the dataset.

        Note: Done in batches to reduce computational complexity.
        """
        train_set = torchvision.datasets.FashionMNIST(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        )

        loader = torch.utils.data.DataLoader(train_set, batch_size=1000, num_workers=1)
        total_pixels = len(train_set) * 28 * 28
        total_sum = 0

        # batch[0] contains an image of shape 28x28
        for batch in train_set:
            total_sum += batch[0].sum()

        mean = total_sum / total_pixels

        sum_of_squared_error = 0
        for batch in train_set:
            sum_of_squared_error += ((batch[0] - mean).pow(2)).sum()

        std = torch.sqrt(sum_of_squared_error / total_pixels)
        return mean, std

    def loader(self, normalize):
        if self.dataset_name == "fashion_mnist":
            if normalize:
                mean, std = self.__calculate_stats()
                self.train_set = torchvision.datasets.FashionMNIST(
                    root="./data",
                    train=True,
                    download=True,
                    transform=transforms.Compose(
                        [transforms.ToTensor(), transforms.Normalize(mean, std),]
                    ),
                )
            # set num_workers to 1; adding more workers does not help
            # since the  workers just queue the next batch
            # processing still remains a bottleneck
            return torch.utils.data.DataLoader(
                self.train_set, self.batch_size, num_workers=self.num_workers
            )

