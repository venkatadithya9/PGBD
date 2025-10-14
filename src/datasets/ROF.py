from torchvision.datasets import ImageFolder
import torchvision.transforms as tfs
from torch.utils.data import (
    DataLoader,
    random_split,
    Subset,
    Sampler,
    ConcatDataset,
    Dataset,
)
import torch


def get_mean_and_std(dataset):
    loader = DataLoader(dataset, batch_size=10, num_workers=0, shuffle=False)
    mean = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(loader.dataset)
    var = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
    std = torch.sqrt(var / (len(loader.dataset) * 224 * 224))
    return mean, std


class TransformWrapper(Dataset):

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)

        return img, label


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ROF(metaclass=Singleton):
    def __init__(
        self,
        base_pth="./data/ROF/data",
        base_pth_pois="./data/ROF/data_sunglasses",
        pratio=0.1,
    ) -> None:
        transforms = tfs.Compose(
            [
                tfs.ToTensor(),
                tfs.Resize((64, 64)),
                tfs.RandomHorizontalFlip(),
                tfs.ToPILImage(),
            ]
        )
        self.base_pth = base_pth
        self.base_pth_pois = base_pth_pois
        self.clean_dataset = ImageFolder(base_pth, transform=None)
        self.pois_dataset = ImageFolder(base_pth_pois, transform=transforms)
        train_data, test_data = random_split(
            self.clean_dataset,
            [
                int(0.9 * len(self.clean_dataset)),
                len(self.clean_dataset) - int(0.9 * len(self.clean_dataset)),
            ],
        )  # doing 90-10 split
        self.clean_train = train_data
        self.clean_test = test_data
        self.num_classes = 10
        self.pratio = pratio
        self.train_data_pois, self.test_data_pois = random_split(
            self.pois_dataset,
            [
                int(self.pratio * (len(self.clean_train))),
                len(self.pois_dataset) - int(self.pratio * (len(self.clean_train))),
            ],
        )
        print(len(self.clean_dataset), len(train_data), len(test_data))

    def get_full_train_data(self):
        train_data = ConcatDataset([self.clean_train, self.train_data_pois])
        return train_data

    def get_clean_train_data(self, transform=None):
        train_data = TransformWrapper(self.clean_train, transform=transform)
        return train_data

    def get_bd_train_data(self, transform=None):
        train_data_pois = TransformWrapper(self.train_data_pois, transform=transform)
        return train_data_pois

    def get_bd_test_data(self, transform=None):
        test_data_pois = TransformWrapper(self.test_data_pois, transform=transform)
        return test_data_pois

    def get_test_data(self, transform=None):
        test_data = TransformWrapper(self.clean_test, transform=transform)
        return test_data
