"""
define different data augmentation for different dataset
time:20201217
"""
import sys

from torchvision import transforms
from transform import Transpose  # Transpose op for EMNIST

from arguments import ExpermentPara


def get_dataset_transform(cur_arguments: ExpermentPara):
    """
    get train and test dataset transform for different dataset
    :param cur_arguments:
    :return:
    """
    if cur_arguments.dataset_name == "MNIST":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif cur_arguments.dataset_name == "CIFAR10":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
    elif cur_arguments.dataset_name == "CIFAR100":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
    elif cur_arguments.dataset_name == "EMNIST":
        train_transform = transforms.Compose([
            Transpose(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_transform = transforms.Compose([
            Transpose(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        print("current dataset name:{} is not right".format(cur_arguments.dataset_name))
        sys.exit(1)
    return train_transform, test_transform


def get_dataset_transform_v1(cur_arguments: ExpermentPara):
    """
    get train and test dataset transform for different dataset
    add data augmentation for cifar100 dataset
    :param cur_arguments:
    :return:
    """
    if cur_arguments.dataset_name == "MNIST":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif cur_arguments.dataset_name == "CIFAR10":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
    elif cur_arguments.dataset_name == "CIFAR100":
        train_transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(32, padding=6),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
    elif cur_arguments.dataset_name == "EMNIST":
        train_transform = transforms.Compose([
            Transpose(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_transform = transforms.Compose([
            Transpose(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        print("current dataset name:{} is not right".format(cur_arguments.dataset_name))
        sys.exit(1)
    return train_transform, test_transform




def get_dataset_transform_v2(cur_arguments: ExpermentPara):
    """
    get train and test dataset transform for different dataset
    add data augmentation for cifar100 dataset
    :param cur_arguments:
    :return:
    """
    if cur_arguments.dataset_name == "MNIST":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif cur_arguments.dataset_name == "CIFAR10":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
    elif cur_arguments.dataset_name == "CIFAR100":
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=6),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
    elif cur_arguments.dataset_name == "EMNIST":
        train_transform = transforms.Compose([
            Transpose(),
            transforms.RandomCrop(28, padding=6),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_transform = transforms.Compose([
            Transpose(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        print("current dataset name:{} is not right".format(cur_arguments.dataset_name))
        sys.exit(1)
    return train_transform, test_transform