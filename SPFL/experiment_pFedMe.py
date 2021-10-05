"""
remove pysyft framework
new version:fed avg
time:20201217
"""

import time
import copy
import numpy as np
import torch

torch.set_default_tensor_type(torch.FloatTensor)
#  set random seed

import torch.nn.functional as F
import torch.optim as optim

from optimizers.fedoptimizer import pFedMeOptimizer

from client.client import Client, client_train_schedule_pFedMe as client_train_schedule
# from client.client import Client, client_train_schedule_fedavg as client_train_schedule

from dataset.org_dataset import load_org_dataset
from dataset.dataset_v4 import dataset_federate_new as data_federate,BaseDataset
# from dataset.data_transform import get_dataset_transform_v1 as get_dataset_transform
from dataset.data_transform import get_dataset_transform

from model.model_factory import get_init_model

#  
from arguments_v1 import ExperimentPara as ExpermentPara
from logger.log_utils import Logger
from aggregation_utils.aggregation_v2 import AggregationModule

from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 0, 'pin_memory': False} if use_cuda else {}

def federate_train_v1(client_list, cur_arguments: ExpermentPara, cur_round,
                      aggregation_module: AggregationModule=None,
                      compression_module = None):
    org_server_model = copy.deepcopy(client_list[0].model)
    org_model_list = []  # 
    for cur_client in client_list:
        cur_client: Client
        copy_model = copy.deepcopy(cur_client.get_model())
        org_model_list.append(copy_model)

    update_model_list = []
    for cur_client in client_list:  # 
        tmp_model = copy.deepcopy(cur_client.get_model())
        tmp_model.train()
        # tmp_optimizer = optim.SGD(tmp_model.parameters(), lr=1e-2, weight_decay=1e-4)
        tmp_optimizer = pFedMeOptimizer(tmp_model.parameters(), lr=cur_arguments.personal_learning_rate, lamda=cur_arguments.lamda)


        tmp_train_data = cur_client.get_train_dataset()
        # tmp_train_data.set_transform_flag(True)
        tmp_dataloader = torch.utils.data.DataLoader(tmp_train_data, batch_size=cur_arguments.batch_size,
                                                     shuffle=True, num_workers=6, pin_memory=False)
        update_model = client_train_schedule(tmp_model, tmp_dataloader, tmp_optimizer, cur_arguments, device)
        update_model_list.append(update_model)  # 

    for index,cur_client in enumerate(client_list):
        cur_client.set_model(update_model_list[index])
    print("after local update,and then test the personal local model.")
    federate_test_v1(client_list,cur_round)

    update_model_list_cp = []
    for tmp_model in update_model_list:
        current_model_local = copy.deepcopy(tmp_model)
        update_model_list_cp.append(current_model_local)

    # 
    final_state_dict_list = aggregation_module.get_aggregation_result(update_model_list_cp,beta=cur_arguments.beta,org_server_model=org_server_model)

    if cur_arguments.use_server:
        print("update by server at rount {}".format(cur_round))
        for model_index, tmp_model in enumerate(update_model_list):  # 
            tmp_model.load_state_dict(final_state_dict_list[model_index])
    else:
        pass
    for update_model, cur_client in zip(update_model_list, client_list):  # 
        cur_client.set_model(copy.deepcopy(update_model))  # 
    return client_list


def federate_test_v1(cur_client_list, cur_round):
    all_acc_list = []
    for cur_client in cur_client_list:
        cur_client: Client
        current_worker_id = cur_client.get_worker_id()
        cur_model = cur_client.get_model()
        cur_model.eval()
        cur_test_dataset = cur_client.get_test_dataset()
        client_test_sample_num = len(cur_test_dataset)
        cur_test_dataloader = torch.utils.data.DataLoader(cur_test_dataset, batch_size=50000,
                                                          shuffle=True, num_workers=6, pin_memory=False)
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(cur_test_dataloader):
                data, target = data.to(device), target.to(device)
                output = cur_model(data)
                current_loss = F.nll_loss(output, target, reduction='sum')

                test_loss = test_loss + current_loss.item()  # sum up batch loss

                pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
                correct = correct + pred.eq(target.view_as(pred)).sum().item()
        test_loss /= client_test_sample_num
        log_info = 'Test_result,Epoch={},Model={},Test set:Average loss={:.4f},Accuracy={}/{} ({:.0f}%)'.format(
            cur_round, current_worker_id, test_loss, correct, client_test_sample_num,
            100. * correct / client_test_sample_num)
        all_acc_list.append(correct / client_test_sample_num)
        print(log_info)
    print("np.mean(all_acc_list)",np.mean(all_acc_list))

if __name__ == "__main__":
    import sys
    savedStdout = sys.stdout  # 
    import argparse

    parse = argparse.ArgumentParser()
    parse.add_argument("--mode", type=str, default="IID")
    parse.add_argument("--client_num", type=int, default=10)
    parse.add_argument("--class_num_for_client", type=int, default=32)
    parse.add_argument("--dataset_name", type=str, default="EMNIST")
    parse.add_argument("--model_name", type=str, default="CNN")
    parse.add_argument("--communication_rounds", type=int, default=100)
    parse.add_argument("--local_epoch", type=int, default=1)
    parse.add_argument("--batch_size", type=int, default=128)
    parse.add_argument("--use_server", type=int, default=1)
    parse.add_argument("--re_stdout", type=int, default=0)
    parse.add_argument("--lamda", type=int, default=15, help="Regularization term")
    parse.add_argument("--personal_learning_rate", type=float, default=0.01,
                        help="Persionalized learning rate to caculate theta aproximately using K steps")
    parse.add_argument("--K", type=int, default=2, help="Computation steps")
    parse.add_argument("--beta", type=float, default=1.0,
                        help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parse.add_argument("--learning_rate", type=float, default=0.01, help="Local learning rate")
    parse.add_argument("--seed", type=int, default=50, help="random seed")
    # 
    args = parse.parse_args()
    if args.re_stdout == 0:
        re_stdout = False
    else:
        re_stdout = True
    cur_arguments = ExpermentPara(mode=args.mode,
                                  client_num=args.client_num,
                                  class_num_for_client=args.class_num_for_client,
                                  dataset_name=args.dataset_name,
                                  model_name=args.model_name,
                                  communication_rounds=args.communication_rounds,
                                  local_epoch=args.local_epoch,
                                  batch_size=args.batch_size,
                                  use_server=(args.use_server == 1),
                                  lamda=args.lamda,
                                  personal_learning_rate=args.personal_learning_rate,
                                  K=args.K,
                                  beta=args.beta,
                                  learning_rate=args.learning_rate,
                                  seed=args.seed)
    current_time = time.localtime(time.time())
    time_stamp = (
        "2021-{}-{}-{}-{}-Release-pFedMe".format(current_time[1], current_time[2], current_time[3],
                                                                      current_time[4]))

    if cur_arguments.dataset_name == "MNIST":
        import config.mnist_config as config
    elif cur_arguments.dataset_name == "CIFAR10":
        import config.cifar10_config as config
    elif cur_arguments.dataset_name == "CIFAR100":
        import config.cifar100_config as config
    elif cur_arguments.dataset_name == "EMNIST":
        import config.emnist_config as config
    else:
        print("current dataset name:{} is not right".format(cur_arguments.dataset_name))
        sys.exit(1)

    arguments_str = cur_arguments.get_str_rep()
    arguments_str = arguments_str + time_stamp
    logger = Logger(log_root_dir=config.LOG_DIR,
                    experment_name=arguments_str,
                    arguements=cur_arguments)
    if re_stdout:
        file_path = logger.run_log_file
        file = open(file_path, 'w+')
        sys.stdout = file  # 
        # print("stra")
    print("start experment----> para is:", flush=True)
    print(arguments_str)

    from torchvision import transforms
    from transform import Transpose  # Transpose op for EMNIST

    #set random seed
    # SEED = 50
    torch.manual_seed(cur_arguments.seed)
    torch.cuda.manual_seed(cur_arguments.seed)
    np.random.seed(cur_arguments.seed)
    print("cur_arguments.seed",cur_arguments.seed)

    #  create workers
    machine_list = []
    for i in range(cur_arguments.client_num):
        machine_list.append(str(i))

    #  create dataset
    train_transform,test_transform = get_dataset_transform(cur_arguments)

    current_dataset = load_org_dataset(config.DATASET_NAME,
                                       config.DATA_ROOT)
    print("load original dataset finished!,the dataset shape is {}".format(np.array(current_dataset.data).shape))

    #  create federate train and test dataset
    train_federate_dataset, test_federate_dataset, client2index_list = \
        data_federate(current_dataset, machine_list,
                      distribution_mode=cur_arguments.mode,
                      class_num_client=cur_arguments.class_num_for_client,
                      dataset_name=cur_arguments.dataset_name)

    # save current dataset assign for each client
    logger.save_client_data_index(client2index_list)
    # logger.load_client_data_index()

    model = get_init_model(cur_arguments)  # 

    # create client
    client_list = []
    for client_index, client_id in enumerate(machine_list):
        cur_model = copy.deepcopy(model)
        cur_model.to(device)
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
    model.to(device)

    # 
    aggregation_module = AggregationModule(cur_arguments)

    for epoch in range(1, cur_arguments.communication_round + 1):
        print("------" * 11)
        start = time.time()
        # return client_list:
        client_list = federate_train_v1(client_list, cur_arguments, epoch, aggregation_module=aggregation_module)
        end = time.time()
        train_time = end - start
        start = time.time()
        # print(epoch, ":finish_one_round:", 20 * '*', "result on train dataset")
        # federate_test(all_model_dict, new_federate_train_dataloader1, epoch, cur_arguments)
        print(epoch, ":finish_one_round:", 20 * '*', "result on test dataset using global model")
        federate_test_v1(client_list, epoch)
        end = time.time()
        test_time = end - start

        start = time.time()
        #  save model checkpoint
        if epoch % 10 == 0:
            for index, cur_client in enumerate(client_list):
                current_model_local = cur_client.get_model()
                logger.save_client_model(current_model_local, index, "round_" + str(epoch))
        end = time.time()
        routine_time = end - start
        print("Time cost is Train time:{:.6f},Test time:{:.6f},Routine time:{:.6f}".format(train_time, test_time,
                                                                                           routine_time))
        sys.stdout.flush()
    if re_stdout:
        sys.stdout = savedStdout  # 
        file.close()
