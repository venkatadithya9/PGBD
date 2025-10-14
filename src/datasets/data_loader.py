from torch import nn
from torchvision import transforms, datasets
from torchvision.transforms import functional as F
from torch.utils.data import (
    random_split,
    DataLoader,
    Dataset,
    SubsetRandomSampler,
    Subset,
)
import copy
import torch
import numpy as np
import time
import random
import cv2
from PIL import Image
from torchvision.datasets import DatasetFolder, ImageFolder
from tqdm import tqdm
from utils import Normalizer
from datasets.GTSRB import GTSRB
from datasets.ROF import ROF
from utils import Denormalizer, Normalizer
import pickle as pkl


def get_train_loader(
    opt, target_transform=None, without_loader=False, dino=False, only_data=False
):
    print("==> Preparing train data..")

    if opt.dataset == "CIFAR10":
        tf_train = transforms.Compose(
            [
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ]
        )
        if without_loader:
            tf_train = transforms.Compose(
                [
                    # transforms.RandomCrop(32, padding=4),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
                    ),
                ]
            )
        if dino:
            tf_train = transforms.Compose(
                [
                    # tfs.Resize(256),
                    transforms.Resize((32, 32)),
                    # tfs.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        trainset = datasets.CIFAR10(
            root="data/CIFAR10", train=True, download=True)
    elif opt.dataset == "CIFAR100":
        tf_train = transforms.Compose(
            [
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ]
        )
        if without_loader:
            tf_train = transforms.Compose(
                [
                    # transforms.RandomCrop(32, padding=4),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]
                    ),
                ]
            )
        if dino:
            tf_train = transforms.Compose(
                [
                    # tfs.Resize(256),
                    transforms.Resize((32, 32)),
                    # tfs.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        trainset = datasets.CIFAR100(
            root="./data/CIFAR100", train=True, download=True)
    elif opt.dataset == "imagenet":
        tf_train = transforms.Compose(
            [
                # transforms.RandomResizedCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]
        )
        trainset = ImageFolder(
            "../../NAD/data/Imagenet20/train", transform=transforms.Resize((224, 224))
        )
    elif opt.dataset == "tinyImagenet":
        tf_train = transforms.Compose(
            [
                # transforms.RandomResizedCrop(64),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]
        )
        if without_loader:
            tf_train = transforms.Compose(
                [
                    # transforms.RandomCrop(32, padding=4),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]
                    ),
                ]
            )
        if dino:
            tf_train = transforms.Compose(
                [
                    # tfs.Resize(256),
                    transforms.Resize((32, 32)),
                    # tfs.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        trainset = ImageFolder(
            "../BackdoorBench/data/tiny/tiny-imagenet-200/train",
            transform=transforms.Resize((64, 64)),
        )
    elif opt.dataset == "gtsrb":

        tmp = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((32, 32)),
                transforms.ToPILImage(),
            ]
        )

        tf_train = transforms.Compose(
            [
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]
        )
        if without_loader:
            tf_train = transforms.Compose(
                [
                    # transforms.RandomCrop(32, padding=4),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
                ]
            )
        if dino:
            tf_train = transforms.Compose(
                [
                    # tfs.Resize(256),
                    transforms.Resize((32, 32)),
                    # tfs.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        # trainset = DatasetFolder(
        #     root='./data/GTSRB/train',  # please replace this with path to your training set
        #     loader=cv2.imread,
        #     extensions=('ppm',),
        #     transform=tmp,
        #     target_transform=None)
        trainset = GTSRB(data_root="./data/GTSRB/", train=True, transform=tmp)
    elif opt.dataset == "ROF":
        tmp = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((64, 64)),
                transforms.ToPILImage(),
            ]
        )
        tf_train = transforms.Compose(
            [
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=torch.FloatTensor([0.6668, 0.5134, 0.4482]),
                    std=torch.FloatTensor([0.0691, 0.0599, 0.0585]),
                ),
            ]
        )
        if without_loader:
            tf_train = transforms.Compose(
                [
                    # transforms.RandomCrop(32, padding=4),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=torch.FloatTensor([0.6668, 0.5134, 0.4482]),
                        std=torch.FloatTensor([0.0691, 0.0599, 0.0585]),
                    ),
                ]
            )
        if dino:
            tf_train = transforms.Compose(
                [
                    # tfs.Resize(256),
                    transforms.Resize((64, 64)),
                    # tfs.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        trainset = ROF().get_clean_train_data(transform=tmp)
    else:
        raise Exception("Invalid dataset")

    if only_data:
        return DatasetCL(
            opt,
            full_dataset=trainset,
            transform=None,
            target_transform=target_transform,
        )

    train_data = DatasetCL(
        opt,
        full_dataset=trainset,
        transform=tf_train,
        target_transform=target_transform,
    )
    if without_loader:
        return train_data
    train_loader = DataLoader(
        train_data, batch_size=opt.batch_size, shuffle=True, drop_last=True
    )

    return train_loader


def get_test_loader(opt):
    print("==> Preparing test data..")
    if opt.dataset == "CIFAR10":
        tf_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        testset = datasets.CIFAR10(
            root="data/CIFAR10", train=False, download=True)
    elif opt.dataset == "CIFAR100":
        tf_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]
                ),
            ]
        )
        testset = datasets.CIFAR100(
            root="data/CIFAR100", train=False, download=True)
    elif opt.dataset == "tinyImagenet":
        tf_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]
                ),
            ]
        )
        testset = ImageFolder(
            "../BackdoorBench/data/tiny/tiny-imagenet-200/val",
            transform=transforms.Resize((64, 64)),
        )
        testset, _ = random_split(
            testset, [int(0.1 * len(testset)), int(0.9 * len(testset))]
        )
    elif opt.dataset == "gtsrb":

        tmp = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((32, 32)),
                transforms.ToPILImage(),
            ]
        )
        tf_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0, 0, 0), (1, 1, 1)),
                # transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669))
            ]
        )

        testset = GTSRB(data_root="data/GTSRB", train=False, transform=tmp)
    elif opt.dataset == "ROF":

        tmp = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((64, 64)),
                transforms.ToPILImage(),
            ]
        )
        tf_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0, 0, 0), (1, 1, 1)),
                # transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669))
            ]
        )

        testset = []
    else:
        raise Exception("Invalid dataset")

    test_data_clean = DatasetBD(
        opt, full_dataset=testset, inject_portion=0, transform=tf_test, mode="test"
    )
    test_data_bad = DatasetBD(
        opt, full_dataset=testset, inject_portion=1, transform=tf_test, mode="test"
    )

    # (apart from label 0) bad test data
    test_clean_loader = DataLoader(
        dataset=test_data_clean,
        batch_size=opt.batch_size,
        shuffle=False,
    )
    # all clean test data
    test_bad_loader = DataLoader(
        dataset=test_data_bad,
        batch_size=opt.batch_size,
        shuffle=False,
    )

    return test_clean_loader, test_bad_loader


def get_backdoor_loader(
    opt,
    shuffle=True,
    batch_size=1,
    use_available=False,
    without_loader=False,
    dino=False,
    pattern=None,
):
    print("==> Preparing train data..")
    if opt.dataset == "CIFAR10":
        tf_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        if without_loader:
            tf_train = transforms.Compose(
                [
                    # transforms.RandomCrop(32, padding=4),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )
        if dino:
            tf_train = transforms.Compose(
                [
                    # tfs.Resize(256),
                    transforms.Resize((32, 32)),
                    # tfs.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        trainset = datasets.CIFAR10(
            root="data/CIFAR10", train=True, download=True)

    elif opt.dataset == "gtsrb":

        tmp = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((32, 32)),
                transforms.ToPILImage(),
            ]
        )

        tf_train = transforms.Compose([transforms.ToTensor()])

        if dino:
            tf_train = transforms.Compose(
                [
                    # tfs.Resize(256),
                    transforms.Resize((32, 32)),
                    # tfs.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        # trainset = DatasetFolder(
        #     root='./data/GTSRB/train',  # please replace this with path to your training set
        #     loader=cv2.imread,
        #     extensions=('ppm',),
        #     transform=tmp,
        #     target_transform=None)
        trainset = GTSRB(data_root="./data/GTSRB/", train=True, transform=tmp)
    elif opt.dataset == "ROF":

        tmp = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((64, 64)),
                transforms.ToPILImage(),
            ]
        )

        tf_train = transforms.Compose([transforms.ToTensor()])

        if dino:
            tf_train = transforms.Compose(
                [
                    # tfs.Resize(256),
                    transforms.Resize((64, 64)),
                    # tfs.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        trainset = ROF().get_clean_train_data(transform=tmp)

    elif opt.dataset == "CIFAR100":
        tf_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]
                ),
            ]
        )
        if without_loader:
            tf_train = transforms.Compose(
                [
                    # transforms.RandomCrop(32, padding=4),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]
                    ),
                ]
            )
        if dino:
            tf_train = transforms.Compose(
                [
                    # tfs.Resize(256),
                    transforms.Resize((32, 32)),
                    # tfs.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        trainset = datasets.CIFAR100(
            root="./data/CIFAR100", train=True, download=True)
    elif opt.dataset == "tinyImagenet":
        tf_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]
                ),
            ]
        )
        if without_loader:
            tf_train = transforms.Compose(
                [
                    # transforms.RandomCrop(32, padding=4),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]
                    ),
                ]
            )
        if dino:
            tf_train = transforms.Compose(
                [
                    # tfs.Resize(256),
                    transforms.Resize((32, 32)),
                    # tfs.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        trainset = ImageFolder(
            "../BackdoorBench/data/tiny/tiny-imagenet-200/train",
            transform=transforms.Resize((64, 64)),
        )
    else:
        raise Exception("Invalid dataset")

    if without_loader:
        train_data_bad = DatasetBD(
            opt,
            full_dataset=trainset,
            inject_portion=1,
            transform=tf_train,
            mode="train",
            use_available=use_available,
        )
        return train_data_bad
    print("Generating test bad loader")
    train_data_bad = DatasetBD(
        opt,
        full_dataset=trainset,
        inject_portion=opt.inject_portion,
        transform=tf_train,
        mode="train",
        use_available=use_available,
        pattern=pattern,
    )
    train_clean_loader = DataLoader(
        dataset=train_data_bad,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        drop_last=True,
    )

    return train_clean_loader


class DatasetCL(Dataset):
    def __init__(self, opt, full_dataset=None, transform=None, target_transform=None):
        self.dataset = self.random_split(
            full_dataset=full_dataset, ratio=opt.ratio)
        self.transform = transform
        self.target_transform = target_transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return self.dataLen

    def random_split(self, full_dataset, ratio):
        print("full_train:", len(full_dataset))
        train_size = int(ratio * len(full_dataset))
        drop_size = len(full_dataset) - train_size
        train_dataset, drop_dataset = random_split(
            full_dataset, [train_size, drop_size]
        )
        print("train_size:", len(train_dataset),
              "drop_size:", len(drop_dataset))

        return train_dataset


class DatasetBD(Dataset):
    def __init__(
        self,
        opt,
        full_dataset,
        inject_portion,
        transform=None,
        mode="train",
        device=torch.device("cuda"),
        distance=0,
        use_available=False,
        pattern=None,
    ):
        self.use_available = use_available
        self.pattern = pattern
        if self.use_available:
            full_dataset = self.random_split(full_dataset, opt.ratio)
            print("USE AVAILABLE: ", full_dataset)
        self.dataset = self.addTrigger(
            dataset=full_dataset,
            target_label=opt.target_label,
            inject_portion=inject_portion,
            mode=mode,
            distance=distance,
            trig_w=opt.trig_w,
            trig_h=opt.trig_h,
            trigger_type=opt.trigger_type,
            target_type=opt.target_type,
            t_dataset=opt.dataset,
        )
        self.device = device
        self.transform = transform

    def __getitem__(self, item):
        img = self.dataset[item][0]
        label = self.dataset[item][1]
        img = Image.fromarray(img)
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.dataset)

    def random_split(self, full_dataset, ratio):
        print("full_train:", len(full_dataset))
        train_size = int(ratio * len(full_dataset))
        drop_size = len(full_dataset) - train_size
        train_dataset, drop_dataset = random_split(
            full_dataset, [train_size, drop_size]
        )
        print("train_size:", len(train_dataset),
              "drop_size:", len(drop_dataset))

        return train_dataset

    def addTrigger(
        self,
        dataset,
        target_label,
        inject_portion,
        mode,
        distance,
        trig_w,
        trig_h,
        trigger_type,
        target_type,
        t_dataset=None,
    ):
        print("Generating " + mode + "bad Imgs")
        if target_type != "cleanLabel" or inject_portion == 1:
            self.perm = np.random.permutation(len(dataset))[
                0: int(len(dataset) * inject_portion)
            ]
        else:
            self.perm = []

        # dataset
        dataset_ = list()

        cnt = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]

            if target_type == "all2one":

                if mode == "train":
                    img = np.array(data[0])
                    width = img.shape[0]  # img.shape[0]
                    height = img.shape[1]  # img.shape[1]
                    # if self.use_available:
                    #     width = img.shape[1]  #img.shape[0]
                    #     height = img.shape[2] #img.shape[1]
                    if i in self.perm:
                        # select trigger
                        img = self.selectTrigger(
                            img,
                            width,
                            height,
                            distance,
                            trig_w,
                            trig_h,
                            trigger_type,
                            t_dataset=t_dataset,
                            idx=i,
                        )
                        # change target
                        if self.use_available:
                            dataset_.append((img, data[1]))
                        else:
                            dataset_.append((img, target_label))
                        # dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))
                else:
                    if data[1] == target_label:
                        continue

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in self.perm:
                        img = self.selectTrigger(
                            img,
                            width,
                            height,
                            distance,
                            trig_w,
                            trig_h,
                            trigger_type,
                            t_dataset=t_dataset,
                            idx=i,
                        )

                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

            # all2all attack
            elif target_type == "all2all":

                if mode == "train":
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in self.perm:

                        img = self.selectTrigger(
                            img,
                            width,
                            height,
                            distance,
                            trig_w,
                            trig_h,
                            trigger_type,
                            t_dataset=t_dataset,
                        )
                        target_ = self._change_label_next(data[1])

                        dataset_.append((img, target_))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

                else:

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in self.perm:
                        img = self.selectTrigger(
                            img,
                            width,
                            height,
                            distance,
                            trig_w,
                            trig_h,
                            trigger_type,
                            t_dataset=t_dataset,
                        )

                        target_ = self._change_label_next(data[1])
                        # print(f"In test all2all, new target {target_}, old target {data[1]}")
                        dataset_.append((img, target_))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

            # clean label attack
            elif target_type == "cleanLabel":

                if mode == "train":
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if random.random() < inject_portion * 10:
                        # if i in perm:
                        if data[1] == target_label:
                            self.perm.append(i)
                            img = self.selectTrigger(
                                img,
                                width,
                                height,
                                distance,
                                trig_w,
                                trig_h,
                                trigger_type,
                                t_dataset=t_dataset,
                                idx=i,
                            )

                            dataset_.append((img, data[1]))
                            cnt += 1

                        else:
                            dataset_.append((img, data[1]))
                    else:
                        dataset_.append((img, data[1]))

                else:
                    if data[1] == target_label:
                        continue

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in self.perm:
                        img = self.selectTrigger(
                            img,
                            width,
                            height,
                            distance,
                            trig_w,
                            trig_h,
                            trigger_type,
                            t_dataset=t_dataset,
                        )

                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

        time.sleep(0.01)
        print(
            "Injecting Over: "
            + str(cnt)
            + "Bad Imgs, "
            + str(len(dataset) - cnt)
            + "Clean Imgs"
        )

        return dataset_

    def _change_label_next(self, label):
        label_new = (label + 1) % 10
        return label_new

    def selectTrigger(
        self,
        img,
        width,
        height,
        distance,
        trig_w,
        trig_h,
        triggerType,
        t_dataset=None,
        idx=0,
    ):

        assert triggerType in [
            "squareTrigger",
            "gridTrigger",
            "randomPixelTrigger",
            "fourCornerTrigger",
            "signalTrigger",
            "trojanTrigger",
            "blendTrigger",
            "centerTrigger",
            "wanetTrigger",
            "inputawareTrigger",
            "semanticTrigger",
            "semanticMaskTrigger",
            "semanticTattooTrigger",
            "combatTrigger",
            "customSquareTrigger",
        ]
        if self.pattern != None:
            print("Synth trigger added")
            img = torch.tensor(img)
            img = torch.clamp(img + self.pattern, 0, 1)
            img = img.numpy()
        elif triggerType == "squareTrigger":
            img = self._squareTrigger(
                img, width, height, distance, trig_w, trig_h)
        elif triggerType == "customSquareTrigger":
            img = self._customSquareTrigger(
                img, width, height, distance, trig_w, trig_h
            )
        elif triggerType == "gridTrigger":
            img = self._gridTriger(
                img, width, height, distance, trig_w, trig_h, t_dataset
            )
        elif triggerType == "fourCornerTrigger":
            img = self._fourCornerTrigger(
                img, width, height, distance, trig_w, trig_h)
        elif triggerType == "centerTrigger":
            img = self._centerTriger(
                img, width, height, distance, trig_w, trig_h)
        elif triggerType == "blendTrigger":
            img = self._blendTrigger(
                img, width, height, distance, trig_w, trig_h, t_dataset
            )
        elif triggerType == "signalTrigger":
            img = self._signalTrigger(
                img, width, height, distance, trig_w, trig_h)
        elif triggerType == "trojanTrigger":
            img = self._trojanTrigger(
                img, width, height, distance, trig_w, trig_h, t_dataset
            )
        elif triggerType == "randomPixelTrigger":
            img = self._randomPixelTrigger(
                img, width, height, distance, trig_w, trig_h)
        elif triggerType == "wanetTrigger":
            # print("SHAPE:   ", img.shape)
            # if train:
            denormalize = Denormalizer("CIFAR10")
            normalize = Normalizer("CIFAR10")
            to_tensor = transforms.ToTensor()

            img = normalize(to_tensor(img).type(torch.float32))
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
            k = 4
            s = 0.5
            grid_rescale = 1
            input_height = 32
            ins = (
                torch.rand(1, 2, k, k) * 2 - 1
            )  # generate (1,2,4,4) shape [-1,1] gaussian
            ins = ins / torch.mean(
                torch.abs(ins)
            )  # scale up, increase var, so that mean of positive part and negative be +1 and -1
            noise_grid = (
                torch.nn.functional.upsample(
                    ins, size=input_height, mode="bicubic", align_corners=True
                )  # here upsample and make the dimension match
                .permute(0, 2, 3, 1)
                .to(device)
            )
            array1d = torch.linspace(-1, 1, steps=input_height)
            x, y = torch.meshgrid(
                array1d, array1d
            )  # form two mesh grid correspoding to x, y of each position in height * width matrix
            identity_grid = torch.stack((y, x), 2)[None, ...].to(
                device
            )  # stack x,y like two layer, then add one more dimension at first place. (have torch.Size([1, 32, 32, 2]))
            grid_temps = (identity_grid + s * noise_grid /
                          input_height) * grid_rescale
            grid_temps = torch.clamp(grid_temps, -1, 1)
            # print(img.shape, grid_temps.shape ,grid_temps.repeat(1, 1, 1, 1).shape)
            img = denormalize(
                torch.nn.functional.grid_sample(
                    img.to(device).unsqueeze(dim=0),
                    grid_temps.repeat(1, 1, 1, 1),
                    align_corners=True,
                )
            )
            # print(img.shape, grid_temps.shape ,grid_temps.repeat(1, 1, 1, 1).shape)
            img = img.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            # print(img.shape)

            # img = img
        elif triggerType == "inputawareTrigger":
            img = img
        elif (
            triggerType == "semanticTrigger"
            or triggerType == "semanticMaskTrigger"
            or triggerType == "semanticTattooTrigger"
            or triggerType == "combatTrigger"
        ):
            img = img
        else:
            raise NotImplementedError

        return img

    def _squareTrigger(self, img, width, height, distance, trig_w, trig_h):

        for j in range(width - distance - trig_w, width - distance):
            for k in range(height - distance - trig_h, height - distance):
                img[j, k] = 255.0

        return img

    def _customSquareTrigger(self, img, width, height, distance, trig_w, trig_h):
        # for j in range(width - distance - trig_w, width - distance):
        #     for k in range(height - distance - trig_h, height - distance):
        #         img[j, k] = 255.0
        # img[width - distance - trig_w - 1: width,height-distance-trig_h-1] = [191, 191, 191]
        # img[width - distance - trig_w - 1,height-distance-trig_h-1: height] = [191, 191, 191]
        with open("badnet_tiny_trig.pkl", "rb") as f:
            trig = pkl.load(f)
        img[width - trig.shape[0]: width, height -
            trig.shape[1]: height, :] = trig
        return img

    def _gridTriger(self, img, width, height, distance, trig_w, trig_h, t_dataset):

        if self.use_available:
            img[:, width - 1, height - 1] = 255
            img[:, width - 1, height - 2] = 0
            img[:, width - 1, height - 3] = 255

            img[:, width - 2, height - 1] = 0
            img[:, width - 2, height - 2] = 255
            img[:, width - 2, height - 3] = 0

            img[:, width - 3, height - 1] = 255
            img[:, width - 3, height - 2] = 0
            img[:, width - 3, height - 3] = 0
        else:
            img[width - 1][height - 1] = 255
            img[width - 1][height - 2] = 0
            img[width - 1][height - 3] = 255

            img[width - 2][height - 1] = 0
            img[width - 2][height - 2] = 255
            img[width - 2][height - 3] = 0

            img[width - 3][height - 1] = 255
            img[width - 3][height - 2] = 0
            img[width - 3][height - 3] = 0

        return img

    def _fourCornerTrigger(self, img, width, height, distance, trig_w, trig_h):
        # right bottom
        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        # left top
        img[1][1] = 255
        img[1][2] = 0
        img[1][3] = 255

        img[2][1] = 0
        img[2][2] = 255
        img[2][3] = 0

        img[3][1] = 255
        img[3][2] = 0
        img[3][3] = 0

        # right top
        img[width - 1][1] = 255
        img[width - 1][2] = 0
        img[width - 1][3] = 255

        img[width - 2][1] = 0
        img[width - 2][2] = 255
        img[width - 2][3] = 0

        img[width - 3][1] = 255
        img[width - 3][2] = 0
        img[width - 3][3] = 0

        # left bottom
        img[1][height - 1] = 255
        img[2][height - 1] = 0
        img[3][height - 1] = 255

        img[1][height - 2] = 0
        img[2][height - 2] = 255
        img[3][height - 2] = 0

        img[1][height - 3] = 255
        img[2][height - 3] = 0
        img[3][height - 3] = 0

        return img

    def _centerTriger(self, img, width, height, distance, trig_w, trig_h):

        # adptive center trigger
        alpha = 1
        img[width - 14][height - 14] = 255 * alpha
        img[width - 14][height - 13] = 0 * alpha
        img[width - 14][height - 12] = 255 * alpha

        img[width - 13][height - 14] = 0 * alpha
        img[width - 13][height - 13] = 255 * alpha
        img[width - 13][height - 12] = 0 * alpha

        img[width - 12][height - 14] = 255 * alpha
        img[width - 12][height - 13] = 0 * alpha
        img[width - 12][height - 12] = 0 * alpha

        return img

    def _randomPixelTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.2
        mask = np.random.randint(
            low=0, high=256, size=(width, height), dtype=np.uint8)
        blend_img = (1 - alpha) * img + alpha * \
            mask.reshape((width, height, 1))
        blend_img = np.clip(blend_img.astype("uint8"), 0, 255)

        # print(blend_img.dtype)
        return blend_img

    def _signalTrigger(self, img, width, height, distance, trig_w, trig_h):
        # alpha = 0.2
        # # load signal mask
        # signal_mask = np.load('trigger/signal_cifar10_mask.npy')
        # blend_img = (1 - alpha) * img + alpha * signal_mask.reshape((width, height, 1))  # FOR CIFAR10
        # blend_img = np.clip(blend_img.astype('uint8'), 0, 255)
        delta = 40
        f = 6
        img = np.float32(img)
        pattern = np.zeros_like(img)
        m = pattern.shape[1]
        for i in range(int(img.shape[0])):
            for j in range(int(img.shape[1])):
                pattern[i, j] = delta * np.sin(2 * np.pi * j * f / m)

        img = np.uint32(img) + pattern
        img = np.uint8(np.clip(img, 0, 255))

        return img

    def _trojanTrigger(self, img, width, height, distance, trig_w, trig_h, t_dataset):
        # if t_dataset == 'gtsrb':
        #     trigger_ptn = Image.open('trigger/trigger_gtsrb.png').convert("RGB")
        #     trigger_ptn = np.array(trigger_ptn)
        #     trigger_loc = np.nonzero(trigger_ptn)
        #     img[trigger_loc] = 0
        #     img_ = img + trigger_ptn
        # elif t_dataset == 'CIFAR10':
        #     # load trojanmask
        #     trg = np.load('trigger/best_square_trigger_cifar10.npz')['x']
        #     trg = np.load(f"trigger/trojannn_preactresnet18_{t_dataset}.npz")
        #     # trg.shape: (3, 32, 32)
        #     # trg = np.transpose(trg, (1, 2, 0))

        # elif t_dataset == 'CIFAR100':
        #     trigger_ptn = Image.open('trigger/trigger_cifar100.png').convert("RGB")
        #     trigger_ptn = np.array(trigger_ptn)
        #     trigger_loc = np.nonzero(trigger_ptn)
        #     img[trigger_loc] = 0
        #     img_ = img + trigger_ptn

        trg = np.load(f"trigger/trojannn_preactresnet18_{t_dataset}.npz")
        img_ = np.clip((img + trg).astype("uint8"), 0, 255)
        return img_

    def _blendTrigger(self, img, width, height, distance, trig_w, trig_h, t_dataset):
        alpha = 0.2
        poison_img = copy.deepcopy(img)
        poison_img = poison_img.astype(np.float64)
        poison_img /= 255
        # img_to_tensor = transforms.PILToTensor()
        trigger_img = Image.open("./trigger/hello_kitty.jpeg")
        trigger_img = trigger_img.resize(
            (height, width), resample=Image.Resampling.BILINEAR
        )
        trigger = np.array(trigger_img).astype(np.float64)
        trigger /= 255
        # print(np.max(trigger), np.max(img))
        poison_img = (1 - alpha) * poison_img + (alpha) * trigger
        # print(img.shape, poison_img.shape)
        # # adptive center trigger
        # poison_img[width - 14][height - 14] = 255 * alpha + (1 - alpha) * poison_img[width - 14][height - 14]
        # poison_img[width - 14][height - 13] = 128 * alpha + (1 - alpha) * poison_img[width - 14][height - 13]
        # poison_img[width - 14][height - 12] = 255 * alpha + (1 - alpha) * poison_img[width - 14][height - 12]

        # poison_img[width - 13][height - 14] = 128 * alpha + (1 - alpha) * poison_img[width - 13][height - 14]
        # poison_img[width - 13][height - 13] = 255 * alpha + (1 - alpha) * poison_img[width - 13][height - 13]
        # poison_img[width - 13][height - 12] = 128 * alpha + (1 - alpha) * poison_img[width - 13][height - 12]

        # poison_img[width - 12][height - 14] = 255 * alpha + (1 - alpha) * poison_img[width - 12][height - 14]
        # poison_img[width - 12][height - 13] = 128 * alpha + (1 - alpha) * poison_img[width - 12][height - 13]
        # poison_img[width - 12][height - 12] = 128 * alpha + (1 - alpha) * poison_img[width - 12][height - 12]
        poison_img *= 255
        return np.array(poison_img).astype(np.uint8)

    # def _wanetTrigger(self, img, width, height, distance, trig_w, trig_h, t_dataset):
