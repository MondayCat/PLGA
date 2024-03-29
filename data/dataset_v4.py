"""
v4:remove pysyft module
time:20201216
code is based on v3
"""

import sys

import numpy as np
import torch
from PIL import Image

sys.path.append("..")
sys.path.append(".")
from data.fede_data_split import FederateDataSplit


class BaseDataset:
    """
    This is a base class to used for manipulating a dataset. This is composed
    of a .data attribute for inputs and a .targets one for labels. It is to
    be used like the MNIST Dataset object, and is useful to avoid handling
    the two inputs and label tensors separately.

    Args:

        data[list,torch tensors]: the data points
        targets: Corresponding labels of the data points
        transform: Function to transform the datapoints

    """

    def __init__(self, data, targets, transform=None):

        self.data = data
        self.targets = targets
        self.transform_ = transform
        self.use_transform = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        """
        Args:

            index[integer]: index of item of to get

        Returns:

            data: Data points corresponding to the given index
            targets: Targets correspoding to given datapoint
        """
        data_elem = self.data[index]
        data_elem = Image.fromarray(data_elem)
        if self.transform_ is not None and self.use_transform:
            # TODO: avoid passing through numpy domain
            data_elem = self.transform_(data_elem)

        return data_elem, self.targets[index]

    def set_transform(self,transform):
        """

        :param transform: transform fun for data ayg
        :return:
        """
        self.transform_ = transform

    def set_transform_flag(self,flag):
        self.use_transform = flag

def dataset_federate_new(dataset, workers, distribution_mode="IID", **kwargs):
    """
    generator federate dataset
    :param dataset:
    :param workers:
    :param distribution_mode:
    :param dataset_name:
    :param kwargs:
    :return:
    """
    #  Fix for old versions of torchvision
    if not hasattr(dataset, "data"):
        if hasattr(dataset, "train_data"):
            dataset.data = dataset.train_data
        elif hasattr(dataset, "test_data"):
            dataset.data = dataset.test_data
        else:
            raise AttributeError("Could not find inputs in dataset")
    if not hasattr(dataset, "targets"):
        if hasattr(dataset, "train_labels"):
            dataset.targets = dataset.train_labels
        elif hasattr(dataset, "test_labels"):
            dataset.targets = dataset.test_labels
        else:
            raise AttributeError("Could not find targets in dataset")
    #  Get Split Result
    num_worker = len(workers)
    data_fede_split = FederateDataSplit(dataset=dataset,
                                        num_worker=num_worker,
                                        mode=distribution_mode,
                                        num_class_client=kwargs["class_num_client"])
    # client2indexs_list = data_fede_split.get_split_result(dataset_name=kwargs["dataset_name"])
    client2indexs_list = data_fede_split.get_split_result()
    #  check result
    print("samples of index for client 0")
    print(client2indexs_list[0][0][:10])

    org_data = np.array(dataset.data)
    org_targets = np.array(dataset.targets)
    print(org_data[0].shape)
    print(org_data.shape,org_targets.shape,type(org_data),type(org_targets))

    train_datasets = []
    test_datasets = []
    for i in range(num_worker):
        cur_index_lists = client2indexs_list[i]
        merger_train_data = []
        merger_train_targets = []
        merger_test_data = []
        merger_test_targets = []
        for cur_index_list in cur_index_lists:
            cur_sample_num = len(cur_index_list)
            cur_train_num = int(cur_sample_num*0.8)

            cur_data = org_data[cur_index_list]
            cur_targets = org_targets[cur_index_list]
            cur_train_data = cur_data[:cur_train_num]
            cur_test_data = cur_data[cur_train_num:]
            cur_train_targets = cur_targets[:cur_train_num]
            cur_test_targets = cur_targets[cur_train_num:]

            merger_train_data.append(cur_train_data)
            merger_test_data.append(cur_test_data)
            merger_train_targets.append(cur_train_targets)
            merger_test_targets.append(cur_test_targets)
        merger_train_data = np.concatenate(merger_train_data, axis=0)
        merger_test_data = np.concatenate(merger_test_data, axis=0)

        merger_train_targets = np.concatenate(merger_train_targets, axis=0)
        merger_test_targets = np.concatenate(merger_test_targets, axis=0)
        if merger_train_data.ndim == 3:
            merger_train_data = np.expand_dims(merger_train_data, -1)
            merger_test_data = np.expand_dims(merger_test_data, -1)
        merger_train_data = torch.Tensor(merger_train_data).permute(0,3,1,2).type(torch.float32)
        merger_train_targets = torch.Tensor(merger_train_targets).type(torch.int64)
        merger_test_data = torch.Tensor(merger_test_data).permute(0, 3, 1, 2).type(torch.float32)
        merger_test_targets = torch.Tensor(merger_test_targets).type(torch.int64)

        print("cur worker id is {},"
              "train data shape is {},"
              "test data shape is {},"
              "merger train targets shape is {}"
              "merget test targets shape is {}".format(i,merger_train_data.shape,
                                                       merger_test_data.shape,
                                                       merger_train_targets.shape,
                                                       merger_test_targets.shape))
        cur_train_pairs = [(x,y) for x,y in zip(merger_train_data,merger_train_targets)]
        cur_test_pairs = [(x,y) for x,y in zip(merger_test_data,merger_test_targets)]
        train_datasets.append(cur_train_pairs)
        test_datasets.append(cur_test_pairs)
    return train_datasets,test_datasets,client2indexs_list


if __name__=="__main__":
    import sys
    sys.path.append("..")
    sys.path.append(".")
    from data.org_dataset import load_org_dataset

    DATASET_NAME = "MNIST"
    DATASET_NAME = "CIFAR10"
    DATASET_NAME = "EMNIST"
    DATASET_NAME = "CIFAR100"
    DATA_ROOT = "/home/yx/Fede_MAML/data/"
    current_dataset = load_org_dataset(DATASET_NAME,
                                       DATA_ROOT)
    print("load original dataset finished!,the dataset shape is {}".format(np.array(current_dataset.data).shape))

    machine_list = []
    for i in range(10):
        machine_list.append([])
    #  create federate train and test dataset
    train_federate_dataset, test_federate_dataset, client2index_list = \
        dataset_federate_new(current_dataset, machine_list,
                      distribution_mode="NIID",
                      class_num_client=6,
                      dataset_name="")



