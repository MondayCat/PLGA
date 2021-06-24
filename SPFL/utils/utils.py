import time
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


def load_org_dataset(dataset_name):
    """
    load org dataset MNIST CIFAR10 CIFAR100 EMNIST torch vision
    :param dataset_name:
    :return:
    """
    assert dataset_name in ["MNIST", "CIFAR10", "CIFAR100", "EMNIST"], "dataset name is not right"

    if dataset_name == "MNIST":
        dataset_mnist = datasets.MNIST("/home/yx/Fede_MAML/data/", train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                    ]))
        current_dataset = dataset_mnist
        return current_dataset
    elif dataset_name == "CIFAR10":
        dataset_CIFAR10 = datasets.CIFAR10("/home/yx/Fede_MAML/data/", train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                                        ]))
        current_dataset = dataset_CIFAR10
        return current_dataset
    elif dataset_name == "CIFAR100":
        dataset_CIFAR100 = datasets.CIFAR100("/home/yx/Fede_MAML/data/", train=True, download=True,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                                          ]))
        current_dataset = dataset_CIFAR100
        return current_dataset

