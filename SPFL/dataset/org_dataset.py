"""
this file for org torch version dataset
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

sys.path.append("..")
from transform import Transpose  # Transpose op for EMNIST


def load_org_dataset(dataset_name, data_cache_root,train=True, *args, **kwargs):
    """
    load org dataset MNIST CIFAR10 CIFAR100 EMNIST torch vision
    :param dataset_name:
    :param data_cache_root:
    :return:
    """
    assert dataset_name in ["MNIST", "CIFAR10", "CIFAR100", "EMNIST"], "dataset name is not right"

    if dataset_name == "MNIST":
        dataset_mnist = datasets.MNIST(data_cache_root, train=train, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,))
                                       ]))
        current_dataset = dataset_mnist
        return current_dataset
    elif dataset_name == "CIFAR10":
        dataset_CIFAR10 = datasets.CIFAR10(data_cache_root, train=train, download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                                           ]))
        current_dataset = dataset_CIFAR10
        return current_dataset
    elif dataset_name == "CIFAR100":
        dataset_CIFAR100 = datasets.CIFAR100(data_cache_root, train=train, download=True,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                                             ]))
        current_dataset = dataset_CIFAR100
        return current_dataset
    elif dataset_name == "EMNIST":
        dataset_emnist = datasets.EMNIST(root=data_cache_root, split="balanced", train=train, download=True,
                                         transform=transforms.Compose([
                                             Transpose(),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))
                                         ]))
        current_dataset = dataset_emnist
        return current_dataset


def load_org_dataset_v1(dataset_name, data_cache_root,train=True, *args, **kwargs):
    """
    load org dataset MNIST CIFAR10 CIFAR100 EMNIST torch vision
    add data aug for cifar100
    :param dataset_name:
    :param data_cache_root:
    :return:

    """
    assert dataset_name in ["MNIST", "CIFAR10", "CIFAR100", "EMNIST"], "dataset name is not right"

    if dataset_name == "MNIST":
        dataset_mnist = datasets.MNIST(data_cache_root, train=train, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,))
                                       ]))
        current_dataset = dataset_mnist
        return current_dataset
    elif dataset_name == "CIFAR10":

        dataset_CIFAR10 = datasets.CIFAR10(data_cache_root, train=train, download=True,
                                           transform=transforms.Compose([
                                               transforms.RandomCrop(32, padding=4),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                                           ]))
        current_dataset = dataset_CIFAR10
        return current_dataset
    elif dataset_name == "CIFAR100":
        dataset_CIFAR100 = datasets.CIFAR100(data_cache_root, train=train, download=True,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                                             ]))
        current_dataset = dataset_CIFAR100
        return current_dataset
    elif dataset_name == "EMNIST":
        dataset_emnist = datasets.EMNIST(root=data_cache_root, split="balanced", train=train, download=True,
                                         transform=transforms.Compose([
                                             Transpose(),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))
                                         ]))
        current_dataset = dataset_emnist
        return current_dataset


"""
test code
"""

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch
    sys.path.append("..")
    from config import cifar10_config
    from config import cifar100_config
    from config import mnist_config
    from config import emnist_config
    # org_data
    cur_dataset = load_org_dataset(cifar10_config.DATASET_NAME,
                                   cifar10_config.DATA_ROOT)

    print(cur_dataset.data.shape)
    print(cur_dataset.targets[:10])

    all_num = len(cur_dataset)
    data_loader = torch.utils.data.DataLoader(cur_dataset, batch_size=all_num)
    for data, targets in data_loader:
        org_data_torch = data
        org_targets_torch = targets
    print(org_data_torch.shape)
    print(org_targets_torch.shape)
    print(org_targets_torch[:10])

    random_indexs = np.random.choice(all_num, 10)
    for r_index in random_indexs:
        cur_label = cur_dataset.targets[r_index]
        org_image = np.array(cur_dataset.data[r_index])
        tor_org_image = np.array(org_data_torch[r_index].permute((1,2,0)))
        print(np.mean(org_image))
        print(np.mean(tor_org_image))
        print(cur_label)
        plt.subplot(121)
        plt.imshow(org_image)
        plt.subplot(122)
        plt.imshow(tor_org_image[:,:])
        plt.show()

#
