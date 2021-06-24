"""
code is based on experiment_cifar10_fedavg_v2_eval.py
eval code for all experiment
time:20201029
"""

import copy
import sys

import numpy as np

import torch

torch.set_default_tensor_type(torch.FloatTensor)
#  set random seed
SEED = 777
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(777)

import torch.nn.functional as F
import torch.optim as optim

from client.client import Client, client_train_schedule, client_test_schedule,client_train_schedule_step

from dataset.org_dataset import load_org_dataset
from dataset.dataset_v4 import dataset_federate_new as data_federate

#  相关的日志
import config.cifar10_config as config
from logger.log_utils_v1 import  Logger

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 0, 'pin_memory': False} if use_cuda else {}


def federate_test_v1(cur_client_list, cur_round,mode="Train"):
    acc_list = []
    for cur_client in cur_client_list:
        cur_client: Client
        current_worker_id = cur_client.get_worker_id()
        cur_model = cur_client.get_model()
        cur_model.eval()
        if mode=="Train":
            cur_test_dataset = cur_client.get_train_dataset()
            client_test_sample_num = len(cur_test_dataset)
            cur_test_dataloader = torch.utils.data.DataLoader(cur_test_dataset, batch_size=client_test_sample_num,
                                                              shuffle=True, num_workers=6, pin_memory=False)
        elif mode == "Test":
            cur_test_dataset = cur_client.get_test_dataset()
            client_test_sample_num = len(cur_test_dataset)
            cur_test_dataloader = torch.utils.data.DataLoader(cur_test_dataset, batch_size=client_test_sample_num,
                                                              shuffle=True, num_workers=6, pin_memory=False)
        test_loss = 0
        correct = 0
        acc = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(cur_test_dataloader):
                data, target = data.to(device), target.to(device)
                output = cur_model(data)
                current_loss = F.nll_loss(output, target, reduction='sum')

                test_loss = test_loss + current_loss.item()  # sum up batch loss

                pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
                correct = correct + pred.eq(target.view_as(pred)).sum().item()
        test_loss /= client_test_sample_num
        log_info = 'Test_result,Epoch={},Model={},Test set:Average loss={:.4f},Accuracy={}/{} ({:.4f}%)'.format(
            cur_round, current_worker_id, test_loss, correct, client_test_sample_num,
            100. * correct / client_test_sample_num)
        acc =  correct/client_test_sample_num
        acc_list.append(acc)
        print(log_info)
    print("Mean Val is:",np.mean(acc_list))


def fine_tune_step(client_list,step=10):
    update_model_list = []
    for cur_client in client_list:
        tmp_model = copy.deepcopy(cur_client.get_model())
        tmp_model.train()
        tmp_optimizer = optim.SGD(tmp_model.parameters(), lr=1e-3, weight_decay=1e-4)
        tmp_train_data = cur_client.get_train_dataset()
        # tmp_train_data.set_transform_flag(True)
        # tmp_dataloader = torch.utils.data.DataLoader(tmp_train_data, batch_size=cur_arguments.batch_size,
        #                                              shuffle=True, num_workers=6, pin_memory=False)

        tmp_dataloader = torch.utils.data.DataLoader(tmp_train_data, batch_size=128,
                                                     shuffle=True, num_workers=6, pin_memory=False)
        update_model = client_train_schedule_step(tmp_model,
                                                  tmp_dataloader,
                                                  tmp_optimizer,
                                                  cur_arguments,
                                                  device,step)
        update_model_list.append(update_model)

    for index,cur_client in enumerate(client_list):
        cur_client.set_model(update_model_list[index])
    return client_list


if __name__ == "__main__":


    # experiment_name = "mode=NIID|client_num=10|class_num=60|dataset_name=CIFAR100|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=1|use_quan=0|use_para_decom=0|compress_ratio=4|2020-12-15-14-34-debug-fed_persional_compress_v7"
    # experiment_name = "mode=NIID|client_num=10|class_num=60|dataset_name=CIFAR100|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=1|use_quan=0|use_para_decom=0|compress_ratio=4|2020-12-17-13-21-debug-fed_persional_compress_v7_1_4"
    # experiment_name = "mode=NIID|client_num=10|class_num=60|dataset_name=CIFAR100|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=0|use_quan=0|use_para_decom=0|compress_ratio=4|2020-12-16-23-59-debug-fed_persional_compress_v7_1_4"

    # experiment_name = "mode=IID|client_num=10|class_num=60|dataset_name=CIFAR100|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=False|2020-12-18-11-10-Release-fedavg-v2"
    # experiment_name = "mode=IID|client_num=10|class_num=60|dataset_name=CIFAR100|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|2020-12-18-11-10-Release-fedavg-v2"
    # experiment_name = "mode=NIID|client_num=10|class_num=60|dataset_name=CIFAR100|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=False|2020-12-18-11-10-Release-fedavg-v2"
    # experiment_name = "mode=NIID|client_num=10|class_num=60|dataset_name=CIFAR100|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|2020-12-18-11-10-Release-fedavg-v2"


    # experiment_name = "mode=IID|client_num=10|class_num=60|dataset_name=CIFAR100|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=0|use_quan=0|use_para_decom=0|compress_ratio=4|2020-12-18-12-36-Release-fed_persional_compress_v9"
    # experiment_name = "mode=IID|client_num=10|class_num=60|dataset_name=CIFAR100|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=1|use_quan=0|use_para_decom=0|compress_ratio=4|2020-12-18-12-36-Release-fed_persional_compress_v9"
    # experiment_name = "mode=NIID|client_num=10|class_num=60|dataset_name=CIFAR100|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=0|use_quan=0|use_para_decom=0|compress_ratio=4|2020-12-18-12-36-Release-fed_persional_compress_v9"
    # experiment_name = "mode=NIID|client_num=10|class_num=60|dataset_name=CIFAR100|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=1|use_quan=0|use_para_decom=0|compress_ratio=4|2020-12-18-12-36-Release-fed_persional_compress_v9"


    # experiment_name = "mode=IID|client_num=10|class_num=60|dataset_name=CIFAR100|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|2020-12-18-12-59-Release-pre-fedavg-v3"
    # experiment_name ="mode=NIID|client_num=10|class_num=60|dataset_name=CIFAR100|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|2020-12-18-12-59-Release-pre-fedavg-v3"

    # experiment_name ='mode=NIID|client_num=10|class_num=32|dataset_name=EMNIST|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=1|use_quan=0|use_para_decom=0|compress_ratio=4|2020-1-26-15-59-Release-fed_persional_compress_v9'


    # experiment_name = 'mode=IID|client_num=10|class_num=6|dataset_name=CIFAR10|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|local_fine_step=1||2020-2-27-21-18-Release-fedavg-update-v1'
    # experiment_name = "mode=IID|client_num=10|class_num=6|dataset_name=CIFAR10|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|2020-2-27-16-14-Release-pre-fedavg-v3"
    # experiment_name = "mode=IID|client_num=10|class_num=6|dataset_name=CIFAR10|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=0|use_quan=0|use_para_decom=0|compress_ratio=4|2020-2-27-16-14-Release-fed_persional_compress_v9"
    # experiment_name = "mode=IID|client_num=10|class_num=6|dataset_name=CIFAR10|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=1|use_quan=0|use_para_decom=0|compress_ratio=4|2020-2-27-16-14-Release-fed_persional_compress_v9"
    # experiment_name = "mode=IID|client_num=10|class_num=32|dataset_name=EMNIST|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=1|use_quan=0|use_para_decom=0|compress_ratio=4|2020-2-27-15-32-Release-fed_persional_compress_v11"
    # experiment_name = "mode=IID|client_num=10|class_num=32|dataset_name=EMNIST|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|2020-2-27-15-14-Release-pre-fedavg-v4"
    # experiment_name = "mode=IID|client_num=10|class_num=32|dataset_name=EMNIST|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|local_fine_step=1||2020-2-27-15-14-Release-fedavg-update-v2"
    # experiment_name = "mode=NIID|client_num=10|class_num=32|dataset_name=EMNIST|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=1|use_quan=0|use_para_decom=0|compress_ratio=4|2020-1-26-12-34-Release-fed_persional_compress_v9"
    # experiment_name = "mode=NIID|client_num=10|class_num=32|dataset_name=EMNIST|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|2020-1-26-12-16-Release-pre-fedavg-v3"
    # experiment_name = "mode=NIID|client_num=10|class_num=32|dataset_name=EMNIST|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|local_fine_step=1||2020-1-26-12-16-Release-fedavg-update-v1"

    # experiment_name = "mode=NIID|client_num=10|class_num=60|dataset_name=CIFAR100|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=1|use_quan=0|use_para_decom=0|compress_ratio=4|2020-2-28-18-13-debug-fed_persional_compress_v9_1"
    # experiment_name = "mode=NIID|client_num=10|class_num=60|dataset_name=CIFAR100|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=1|use_quan=0|use_para_decom=0|compress_ratio=4|2020-2-28-18-13-debug-fed_persional_compress_v9_2"
    # experiment_name = "mode=NIID|client_num=10|class_num=60|dataset_name=CIFAR100|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=1|use_quan=0|use_para_decom=0|compress_ratio=4|2020-2-28-11-13-Release-fed_persional_compress_v9"
    # experiment_name = "mode=NIID|client_num=10|class_num=60|dataset_name=CIFAR100|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|2020-2-28-11-13-Release-pre-fedavg-v3"

    # experiment_name = 'mode=IID|client_num=10|class_num=6|dataset_name=MNIST|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=1|use_quan=0|use_para_decom=0|compress_ratio=4|2020-3-2-10-32-debug-fed_persional_compress_v9_3'
    # experiment_name = 'mode=NIID|client_num=10|class_num=6|dataset_name=MNIST|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=1|use_quan=0|use_para_decom=0|compress_ratio=4|2020-3-2-9-59-debug-fed_persional_compress_v9_3'
    # experiment_name = 'mode=IID|client_num=10|class_num=32|dataset_name=EMNIST|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=1|use_quan=0|use_para_decom=0|compress_ratio=4|2020-3-2-10-32-debug-fed_persional_compress_v9_3'
    # experiment_name= 'mode=NIID|client_num=10|class_num=32|dataset_name=EMNIST|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=1|use_quan=0|use_para_decom=0|compress_ratio=4|2020-3-2-9-59-debug-fed_persional_compress_v9_3'
    # experiment_name = 'mode=IID|client_num=10|class_num=6|dataset_name=CIFAR10|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=1|use_quan=0|use_para_decom=0|compress_ratio=4|2020-3-2-10-32-debug-fed_persional_compress_v9_3'
    # experiment_name = 'mode=NIID|client_num=10|class_num=6|dataset_name=CIFAR10|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=1|use_quan=0|use_para_decom=0|compress_ratio=4|2020-3-2-9-59-debug-fed_persional_compress_v9_3'
    # experiment_name = 'mode=IID|client_num=10|class_num=60|dataset_name=CIFAR100|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=1|use_quan=0|use_para_decom=0|compress_ratio=4|2020-3-2-10-54-debug-fed_persional_compress_v9_3'
    # experiment_name = 'mode=NIID|client_num=10|class_num=60|dataset_name=CIFAR100|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=1|use_quan=0|use_para_decom=0|compress_ratio=4|2020-3-2-10-54-debug-fed_persional_compress_v9_3'
    #
    #
    # experiment_name = "mode=IID|client_num=10|class_num=6|dataset_name=CIFAR10|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=0|use_quan=0|use_para_decom=0|compress_ratio=4|2020-3-4-23-7-debug-fed_persional_compress_v9_3"
    # experiment_name = "mode=NIID|client_num=10|class_num=6|dataset_name=CIFAR10|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=0|use_quan=0|use_para_decom=0|compress_ratio=4|2020-3-4-23-27-debug-fed_persional_compress_v9_3"
    # experiment_name = "mode=IID|client_num=10|class_num=60|dataset_name=CIFAR100|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=0|use_quan=0|use_para_decom=0|compress_ratio=4|2020-3-4-23-7-debug-fed_persional_compress_v9_3"
    # experiment_name = "mode=NIID|client_num=10|class_num=60|dataset_name=CIFAR100|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=0|use_quan=0|use_para_decom=0|compress_ratio=4|2020-3-4-23-27-debug-fed_persional_compress_v9_3"
    # experiment_name = "mode=IID|client_num=10|class_num=6|dataset_name=MNIST|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=0|use_quan=0|use_para_decom=0|compress_ratio=4|2020-3-4-23-7-debug-fed_persional_compress_v9_3"
    # experiment_name = "mode=NIID|client_num=10|class_num=6|dataset_name=MNIST|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=0|use_quan=0|use_para_decom=0|compress_ratio=4|2020-3-4-23-27-debug-fed_persional_compress_v9_3"
    # experiment_name = "mode=IID|client_num=10|class_num=32|dataset_name=EMNIST|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=0|use_quan=0|use_para_decom=0|compress_ratio=4|2020-3-4-23-7-debug-fed_persional_compress_v9_3"
    # experiment_name = "mode=NIID|client_num=10|class_num=32|dataset_name=EMNIST|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=0|use_quan=0|use_para_decom=0|compress_ratio=4|2020-3-4-23-27-debug-fed_persional_compress_v9_3"
    # experiment_name = "mode=IID|client_num=10|class_num=60|dataset_name=CIFAR100|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=1|use_quan=0|use_para_decom=0|compress_ratio=4|2020-3-2-10-54-debug-fed_persional_compress_v9_3"
    # experiment_name = "mode=NIID|client_num=10|class_num=60|dataset_name=CIFAR100|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=1|use_quan=0|use_para_decom=0|compress_ratio=4|2020-3-2-10-54-debug-fed_persional_compress_v9_3"
    # experiment_name = "mode=IID|client_num=10|class_num=6|dataset_name=CIFAR10|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=1|use_quan=0|use_para_decom=0|compress_ratio=4|2020-3-2-10-32-debug-fed_persional_compress_v9_3"
    # experiment_name = "mode=NIID|client_num=10|class_num=6|dataset_name=CIFAR10|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=1|use_quan=0|use_para_decom=0|compress_ratio=4|2020-3-2-9-59-debug-fed_persional_compress_v9_3"
    # experiment_name = "mode=IID|client_num=10|class_num=32|dataset_name=EMNIST|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=1|use_quan=0|use_para_decom=0|compress_ratio=4|2020-3-2-10-32-debug-fed_persional_compress_v9_3"
    # experiment_name = "mode=NIID|client_num=10|class_num=32|dataset_name=EMNIST|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=1|use_quan=0|use_para_decom=0|compress_ratio=4|2020-3-2-9-59-debug-fed_persional_compress_v9_3"
    # experiment_name = "mode=NIID|client_num=10|class_num=6|dataset_name=MNIST|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=1|use_quan=0|use_para_decom=0|compress_ratio=4|2020-3-2-9-59-debug-fed_persional_compress_v9_3"
    experiment_name = "mode=IID|client_num=10|class_num=6|dataset_name=MNIST|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=1|use_quan=0|use_para_decom=0|compress_ratio=4|2020-3-2-10-32-debug-fed_persional_compress_v9_3"
    """
    'mode=NIID|client_num=10|class_num=32|dataset_name=EMNIST|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|2020-1-26-12-16-Release-pre-fedavg-v3'
    'mode=NIID|client_num=10|class_num=32|dataset_name=EMNIST|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|local_fine_step=1||2020-1-26-12-16-Release-fedavg-update-v1'
    'mode=NIID|client_num=10|class_num=32|dataset_name=EMNIST|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=0|use_quan=0|use_para_decom=0|compress_ratio=4|2020-1-26-12-34-Release-fed_persional_compress_v9'
    'mode=NIID|client_num=10|class_num=32|dataset_name=EMNIST|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=1|use_quan=0|use_para_decom=0|compress_ratio=4|2020-1-26-12-34-Release-fed_persional_compress_v9'
    
    'mode=NIID|client_num=10|class_num=32|dataset_name=EMNIST|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|2020-1-26-12-16-Release-pre-fedavg-v3'
    'mode=NIID|client_num=10|class_num=32|dataset_name=EMNIST|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|2020-1-26-15-40-Release-pre-fedavg-v3'
    'mode=NIID|client_num=10|class_num=32|dataset_name=EMNIST|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|local_fine_step=1||2020-1-26-12-16-Release-fedavg-update-v1'
    'mode=NIID|client_num=10|class_num=32|dataset_name=EMNIST|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|local_fine_step=1||2020-1-26-15-40-Release-fedavg-update-v1'
    'mode=NIID|client_num=10|class_num=32|dataset_name=EMNIST|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=0|use_quan=0|use_para_decom=0|compress_ratio=4|2020-1-26-12-34-Release-fed_persional_compress_v9'
    'mode=NIID|client_num=10|class_num=32|dataset_name=EMNIST|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=0|use_quan=0|use_para_decom=0|compress_ratio=4|2020-1-26-15-59-Release-fed_persional_compress_v9'
    'mode=NIID|client_num=10|class_num=32|dataset_name=EMNIST|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=1|use_quan=0|use_para_decom=0|compress_ratio=4|2020-1-26-12-34-Release-fed_persional_compress_v9'
    'mode=NIID|client_num=10|class_num=32|dataset_name=EMNIST|model_name=CNN|round=100|local_epoch=1|batch_size=128|use_sever=True|two_update_step=1|use_quan=0|use_para_decom=0|compress_ratio=4|2020-1-26-15-59-Release-fed_persional_compress_v9'

    """
    if "name=MNIST" in experiment_name:
        import config.mnist_config as config
    elif "CIFAR10|model_name" in experiment_name:
        import config.cifar10_config as config
    elif "CIFAR100|model_name" in experiment_name:
        import config.cifar100_config as config
    elif "name=EMNIST" in experiment_name:
        import config.emnist_config as config

    # config.LOG_DIR = "/home/yx/Fede_MAML/MAML_Fede_Release_20200921/log_dir/cifar10"
    logger = Logger(log_root_dir=config.LOG_DIR,
                    experment_name=experiment_name,
                    arguements=None)
    cur_arguments = logger.arguements
    arguments_str = cur_arguments.get_str_rep()

    print("log experment----> para is:")
    print(arguments_str)

    from torchvision import transforms
    from transform import Transpose  # Transpose op for EMNIST

    #  create workers
    machine_list = []
    for i in range(cur_arguments.client_num):
        machine_list.append(str(i))

    #  create dataset
    if cur_arguments.dataset_name == "MNIST":
        train_transform =  transforms.Compose([
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
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        train_transform = transforms.Compose([
            # transforms.RandomVerticalFlip(),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(32, padding=6),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        # train_transform = transforms.Compose([
        #     # transforms.RandomVerticalFlip(),
        #     # transforms.RandomHorizontalFlip(),
        #     # transforms.RandomResizedCrop(32),
        #     # transforms.RandomRotation(30),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        # ])
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

    current_dataset = load_org_dataset(config.DATASET_NAME,
                                       config.DATA_ROOT)
    print("load original dataset finished!,the dataset shape is {}".format(np.array(current_dataset.data).shape))

    #  create federate train and test dataset
    train_federate_dataset, test_federate_dataset, client2index_list = \
        data_federate(current_dataset, machine_list,
                                     distribution_mode=cur_arguments.mode,
                                     class_num_client=cur_arguments.class_num_for_client,
                                     dataset_name=cur_arguments.dataset_name)
    #  save current dataset assign for each client
    # logger.save_client_data_index(client2index_list)
    logger.load_client_data_index()


    # create model
    if "name=MNIST" in  experiment_name:
        from model.cnn_model import Net_mnist_v2 as Net
        model = Net(output_class=10)
    elif "name=EMNIST" in experiment_name:
        from model.cnn_model import Net_mnist_v2 as Net
        model = Net(output_class=47)
    elif "CIFAR10|model_name" in experiment_name:
        from model.cnn_model import Net_cifar_v1 as Net
        model = Net(output_class=10)
    elif "CIFAR100|model_name" in experiment_name:
        from model.cnn_model import Net_cifar_v1 as Net
        model = Net(output_class=100)
    model.to(device)
    print("create model finish")
    print(model)

    # create client
    client_list = []
    for client_index, client_id in enumerate(machine_list):
        cur_model = copy.deepcopy(model)
        cur_model.to(device)
        cur_model = logger.load_client_model(cur_model, client_index, "round_100")
        cur_train_dataset = train_federate_dataset[client_index]
        cur_test_dataset = test_federate_dataset[client_index]
        cur_train_dataset.set_transform(transform=train_transform)
        cur_test_dataset.set_transform(transform=test_transform)
        cur_train_dataset.set_transform_flag(True)
        cur_test_dataset.set_transform_flag(True)
        cur_client = Client(model=cur_model,
                            train_dataset=cur_train_dataset,
                            test_dataset=cur_test_dataset,
                            worker_id=client_id)
        client_list.append(cur_client)
        # print(id(model))
        # print(id(cur_model))

    client_list = fine_tune_step(client_list,step=100)

    epoch = 100
    print(epoch, ":finish_one_round:", 20 * '*', "result on test dataset")
    federate_test_v1(client_list, epoch,mode="Test")
    print(epoch, ":finish_one_round:", 20 * '*', "result on train dataset")
    federate_test_v1(client_list, epoch,mode="Train")
