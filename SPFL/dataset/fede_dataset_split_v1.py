"""
split org dataset to severs part
"""
from collections import defaultdict

import torch
import numpy as np


class FederateDataSplit:

    def __init__(self, dataset, num_worker=10, num_class_client=10, mode="IID"):
        assert mode in ["IID", "NIID"], print("mode should be in IID and NIID")
        assert num_class_client%2==0 and num_class_client!=0, \
            print("the num of client should be even and can not be zeros")
        self.dataset = dataset
        self.num_worker = num_worker
        self.num_class_client = num_class_client
        self.mode = mode

    def get_split_result(self):
        """
        split according to different mode and parameters
        :return:
        """
        all_data_num = len(self.dataset)
        data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=all_data_num)
        # get org data and targets
        for data, targets in data_loader:
            org_data = data.clone().detach()
            org_targets = targets.clone().detach()
        num_class = np.array(torch.max(org_targets).cpu() + 1)
        num_worker = self.num_worker

        #  get per class index
        class2index = {}
        for i in range(num_class):
            index_list = np.where(np.array(org_targets.cpu()) == i)[0]
            print("org data label {} sample num is {}".format(i,len(index_list)))
            r_index_list = np.random.choice(len(index_list), len(index_list), replace=False)
            index_list = [index_list[r_index] for r_index in r_index_list]
            class2index[i] = index_list

        # get select class id
        select_class_list = []
        class2start_index = dict(zip(range(num_class), [0 for i in range(num_class)]))
        if self.mode == "IID":
            cur_r = np.random.choice(num_class,self.num_class_client,replace=False)
            select_class_list = [cur_r for i in range(self.num_worker)]
        elif self.mode == "NIID":
            for i in range(self.num_worker):
                cur_r = np.random.choice(num_class,self.num_class_client,replace=False)
                select_class_list.append(cur_r)
        # split one class num according to different num of client
        all_sample_data_count = 0
        all_sample_data_index = []
        client2data_index = defaultdict(list)
        for i in range(self.num_worker):
            cur_select_class_ids = select_class_list[i]
            for cur_select_class_id in cur_select_class_ids:
                cur_class_index_list = class2index[cur_select_class_id]
                cur_class_num = len(cur_class_index_list)
                bucket_num = int(cur_class_num/num_worker)
                cur_s_index = class2start_index[cur_select_class_id]
                cur_bucket_indexs = cur_class_index_list[cur_s_index:cur_s_index+bucket_num]
                client2data_index[i].append(cur_bucket_indexs)
                class2start_index[cur_select_class_id] = cur_s_index+bucket_num

                all_sample_data_index.extend(cur_bucket_indexs)
                all_sample_data_count += len(cur_class_index_list[cur_s_index:cur_s_index+bucket_num])
        assert len(set(all_sample_data_index))==all_sample_data_count,\
            print("different client may have overloop")
        print("current data generate result is")
        print("num of worker is {}".format(self.num_worker))
        print("num of class  is {}".format(num_class))
        print("num of class for one client is {}".format(self.num_class_client))
        for i in range(self.num_worker):
            print("client id is {}".format(i))
            cur_select_d_index_lists = client2data_index[i]
            for cur_index_list in cur_select_d_index_lists:
                self.array = np.array(targets[cur_index_list].cpu())
                cur_label_list = self.array
                cur_label_set = set(cur_label_list)
                assert len(cur_label_set) == 1,print("class num in one index list must be one")
                print("->has class {},sample class num is {}".format(cur_label_list[0],len(cur_index_list)))
        # check no overloop between different client
        return client2data_index


#  test result
if __name__=="__main__":
    import sys
    import matplotlib.pyplot as plt
    import torch

    from dataset.org_dataset import load_org_dataset
    sys.path.append("..")
    from config import cifar10_config
    from config import cifar100_config
    from config import mnist_config
    from config import emnist_config

    cur_dataset = load_org_dataset(mnist_config.DATASET_NAME,
                                   mnist_config.DATA_ROOT)

    cur_data_split = FederateDataSplit(cur_dataset,10,10,"IID")
    cur_data_split.get_split_result()

    cur_data_split = FederateDataSplit(cur_dataset, 10, 10, "NIID")
    cur_data_split.get_split_result()

    cur_dataset = load_org_dataset(emnist_config.DATASET_NAME,
                                   emnist_config.DATA_ROOT)

    cur_data_split = FederateDataSplit(cur_dataset, 10, 10, "IID")
    cur_data_split.get_split_result()

    cur_data_split = FederateDataSplit(cur_dataset, 10, 10, "NIID")
    cur_data_split.get_split_result()

    cur_dataset = load_org_dataset(cifar10_config.DATASET_NAME,
                                   cifar10_config.DATA_ROOT)

    cur_data_split = FederateDataSplit(cur_dataset, 10, 10, "IID")
    cur_data_split.get_split_result()

    cur_data_split = FederateDataSplit(cur_dataset, 10, 10, "NIID")
    cur_data_split.get_split_result()

    cur_dataset = load_org_dataset(cifar100_config.DATASET_NAME,
                                   cifar100_config.DATA_ROOT)

    cur_data_split = FederateDataSplit(cur_dataset, 10, 10, "IID")
    cur_data_split.get_split_result()

    cur_data_split = FederateDataSplit(cur_dataset, 20, 20, "NIID")
    cur_data_split.get_split_result()

