"""
remove pysyft framework
new version:support data augmentation for dataset cifar100
add some trick for cifar100 dataset

aggregation module is aggregation v4_1_1

freeze last layer
time:
"""

import time
import copy
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

from client.client import Client, client_train_schedule, client_test_schedule

from dataset.org_dataset import load_org_dataset
from dataset.dataset_v4 import dataset_federate_new as data_federate,BaseDataset
from dataset.data_transform import get_dataset_transform_v1 as get_dataset_transform

from model.model_factory import get_init_model

#  
from arguments_v1 import ExperimentParaCompressWithPersonal as ExpermentPara
from logger.log_utils import Logger
from aggregation_utils.aggregation_v4_1_1 import AggregationModule
from compress_utils.compress_v3 import CompressModule, compress_fun

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 0, 'pin_memory': False} if use_cuda else {}


def get_send_parameters(model_list_before_update,
                        model_list_after_update,
                        *args,
                        **kwargs):
    """
    get model parameters that will to be send to server
    could be org model parameters or model grad parameters
    :param model_list_before_update: client model before local update
    :param model_list_after_update: client model after local update
    :param args
    :param kwargs
    :return: a dict that save model location to send parameters
    """

    # aggregation_utils
    update_model_dict = {}  # model para that need to be send to server
    model_dict = kwargs["model_dict_after_update"]
    # get all model paras from client
    for model_index, tmp_update_model in enumerate(model_list_after_update):
        data_location_str = str(tmp_update_model.location).split(' ')[1]
        tmp_update_model = model_dict[data_location_str]
        current_model_remote = copy.deepcopy(tmp_update_model)
        current_model_local = current_model_remote.get()
        #  get grad update
        org_model_local = model_list_before_update[model_index]
        grad_state_dict = copy.deepcopy(current_model_local.state_dict())
        state_dict_aupdated = current_model_local.state_dict()
        state_dict_bupdated = org_model_local.state_dict()
        for key_name in grad_state_dict.keys():
            grad_state_dict[key_name] = state_dict_aupdated[key_name] \
                                        - state_dict_bupdated[key_name]
        current_model_local.load_state_dict(grad_state_dict)
        update_model_dict[data_location_str] = current_model_local

    return update_model_dict


def get_send_parameters_v1(model_list_before_update,
                           model_list_after_update):
    send_parameters_list = []
    for model_before_update, model_after_update in zip(model_list_before_update,
                                                       model_list_after_update):
        cur_model_local = copy.deepcopy(model_before_update)
        grad_state_dict = copy.deepcopy(model_before_update.state_dict())
        state_dict_aupdated = model_after_update.state_dict()
        state_dict_bupdated = model_before_update.state_dict()
        for key_name in grad_state_dict.keys():
            grad_state_dict[key_name] = state_dict_aupdated[key_name] \
                                        - state_dict_bupdated[key_name]
        cur_model_local.load_state_dict(grad_state_dict)
        send_parameters_list.append(cur_model_local)
    return send_parameters_list


def federate_train_v1(client_list, cur_arguments: ExpermentPara, cur_round,
                      aggregation_module: AggregationModule=None,
                      compression_module: CompressModule = None):
    org_model_list = []
    for cur_client in client_list:
        cur_client: Client
        copy_model = copy.deepcopy(cur_client.get_model())
        org_model_list.append(copy_model)
    # make sure model on server and client is same
    aggregation_module
    server_model_list = []

    update_model_list = []
    for cur_client in client_list:
        tmp_model = copy.deepcopy(cur_client.get_model())
        tmp_model.train()
        tmp_optimizer = optim.SGD(tmp_model.parameters(), lr=1e-2, weight_decay=1e-4)
        tmp_train_data = cur_client.get_train_dataset()
        tmp_dataloader = torch.utils.data.DataLoader(tmp_train_data, batch_size=cur_arguments.batch_size,
                                                     shuffle=True, num_workers=6, pin_memory=False)
        update_model = client_train_schedule(tmp_model, tmp_dataloader, tmp_optimizer, cur_arguments, device)
        update_model_list.append(update_model)

    for index,cur_client in enumerate(client_list):
        cur_client.set_model(update_model_list[index])
    federate_test_v1(client_list,cur_round)

    send_para_list = get_send_parameters_v1(org_model_list, update_model_list)
    client_loss_list = [0] * len(client_list)  # loss list
    # get all model paras from client
    print("after local upate weight result is ")
    local_state_dict = update_model_list[0].state_dict()
    print(local_state_dict["conv1.weight"][0, 0, 0, :10])
    # import pdb
    # pdb.set_trace()

    # old version
    # add compress module operator before send it to center sever
    import time
    from multiprocessing import Pool
    recon_para_call_list = []  # list that save AsyncResult object for getting real return value
    recon_para_list_r = []  # list that save real return value
    start = time.time()
    with Pool(4) as pool:
        for i in range(len(send_para_list)):
            one_send_para_dict = send_para_list[i].state_dict()
            for k, v in one_send_para_dict.items():
                one_send_para_dict[k] = v.cpu().numpy()
            recon_para_call_list.append(pool.apply_async(compress_fun,
                                                         args=(compression_module, one_send_para_dict,)))
        pool.close()
        pool.join()
        for n_index in range(len(send_para_list)):
            recon_para_list_r.append(recon_para_call_list[n_index].get())
    #  convert numpy data type to torch data type
    #  reload it to send para and get
    state_dict_list_aupdate_acom = []
    for i in range(len(send_para_list)):
        recon_grad_dict = recon_para_list_r[i]
        for k, v in recon_grad_dict.items():
            recon_grad_dict[k] = torch.tensor(v).type(torch.FloatTensor).to(device)
        send_para_list[i].load_state_dict(recon_grad_dict)
        org_model_local = org_model_list[i]
        state_dict_bupdate = org_model_local.state_dict()
        state_dict_aupdate_acom = copy.deepcopy(state_dict_bupdate)
        for key_name in state_dict_bupdate.keys():
            state_dict_aupdate_acom[key_name] = state_dict_bupdate[key_name] + recon_grad_dict[key_name]
        state_dict_list_aupdate_acom.append(state_dict_aupdate_acom)
    end = time.time()
    print("compress module use time :{}".format(end - start))


    # for index in range(len(client_list)):
    #     print("index-----",index)
    #     recon_dict = state_dict_list_aupdate_acom[i]
    #     org_dict = update_model_list[i].state_dict()
    #     for key_name in org_dict.keys():
    #         print(key_name,"error is",torch.max(torch.abs(recon_dict[key_name]-org_dict[key_name])))

    for index,cur_model in enumerate(update_model_list):
        cur_model.load_state_dict(state_dict_list_aupdate_acom[index])

    for index, cur_client in enumerate(client_list):
        cur_client.set_model(update_model_list[index])

    # federate_test_v1(client_list, cur_round)
    # exit()

    #  receive grad from server and recon new update weight para based on it
    final_state_dict_list = aggregation_module.get_aggregation_result_v4(send_para_list,
                                                                      client_loss_list)
    if cur_arguments.use_server:
        print("update by server at rount {}".format(cur_round))
        for model_index, tmp_model in enumerate(update_model_list):
            current_model_local = copy.deepcopy(tmp_model)
            gard_state_dict = final_state_dict_list[model_index]
            state_dict_updated = current_model_local.state_dict()
            for key_name in gard_state_dict.keys():
                state_dict_updated[key_name] = state_dict_updated[key_name] + gard_state_dict[key_name]
            tmp_model.load_state_dict(state_dict_updated)
    else:
        pass
    for update_model, cur_client in zip(update_model_list, client_list):
        cur_client.set_model(copy.deepcopy(update_model))
    return client_list


def routine_set_v1(client_list,
                   cur_arguments: ExpermentPara,
                   aggregation_module: AggregationModule = None,
                   compression_module=None):
    update_model_list = []
    for cur_client in client_list:
        copy_model = copy.deepcopy(cur_client.get_model())
        update_model_list.append(copy_model)

    aggregation_module.set_init_weight_matrix()
    final_state_dict_list = aggregation_module.get_aggregation_result_mean()

    for i, (update_model, cur_client) in \
            enumerate(zip(update_model_list, client_list)):
        update_model.load_state_dict(final_state_dict_list[i])
        if i == 0:
            init_model = copy.deepcopy(update_model)

        cur_client_model = copy.deepcopy(cur_client.get_model())
        cur_client_model.load_state_dict(final_state_dict_list[i])
        cur_client.set_model(cur_client_model)

    aggregation_module.set_init_model(init_model)

    init_set_v1(client_list, cur_arguments, aggregation_module, compress_module)

    return client_list


def init_set_v1(client_list,
                cur_arguments: ExpermentPara,
                aggregation_module: AggregationModule = None,
                compression_module=None):
    model_list_copy = []
    for cur_client in client_list:
        copy_model = copy.deepcopy(cur_client.get_model())
        model_list_copy.append(copy_model)

    update_model_list = []
    for tmp_model, cur_client in zip(model_list_copy, client_list):
        cur_client: Client
        tmp_model.train()
        tmp_optimizer = optim.SGD(tmp_model.parameters(), lr=1e-2, weight_decay=1e-4)
        tmp_train_data = cur_client.get_train_dataset()
        tmp_dataloader = torch.utils.data.DataLoader(tmp_train_data, batch_size=cur_arguments.batch_size,
                                                     shuffle=True, num_workers=6, pin_memory=False)
        update_model = client_train_schedule(tmp_model, tmp_dataloader, tmp_optimizer, cur_arguments, device)
        update_model_list.append(update_model)

    aggregation_module.update_weight_matrix(update_model_list)


def federate_test_v1(cur_client_list, cur_round):
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
        log_info = 'Test_result,Epoch={},Model={},Test set:Average loss={:.4f},Accuracy={}/{} ({:.4f}%)'.format(
            cur_round, current_worker_id, test_loss, correct, client_test_sample_num,
            100. * correct / client_test_sample_num)

        print(log_info)


if __name__ == "__main__":
    import sys

    savedStdout = sys.stdout  # 

    import argparse

    parse = argparse.ArgumentParser()
    parse.add_argument("--mode", type=str, default="IID")
    parse.add_argument("--client_num", type=int, default=10)
    parse.add_argument("--class_num_for_client", type=int, default=10)
    parse.add_argument("--dataset_name", type=str, default="CIFAR10")
    parse.add_argument("--model_name", type=str, default="CNN")
    parse.add_argument("--communication_rounds", type=int, default=100)
    parse.add_argument("--local_epoch", type=int, default=1)
    parse.add_argument("--batch_size", type=int, default=128)
    parse.add_argument("--use_server", type=int, default=0)
    parse.add_argument("--two_local_update", type=int, default=0)
    parse.add_argument("--use_quan", type=int, default=0)
    parse.add_argument("--use_para_decom", type=int, default=0)
    parse.add_argument("--com_ratio", type=int, default=4)
    parse.add_argument("--re_stdout", type=int, default=0)
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
                                  local_two_update=args.two_local_update,
                                  use_quan=args.use_quan,
                                  use_para_decom=args.use_para_decom)
    current_time = time.localtime(time.time())
    time_stamp = ("2020-{}-{}-{}-{}-debug-fed_persional_compress_v9_3_2".format(current_time[1], current_time[2], current_time[3],current_time[4]))

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

    model = get_init_model(cur_arguments)

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

    # set agg module and compress module
    share_model = copy.deepcopy(model)
    aggregation_module = AggregationModule(cur_arguments, share_model)
    init_model_list = []
    for cur_client in client_list:
        client_model_cp = copy.deepcopy(cur_client.get_model())
        init_model_list.append(client_model_cp)
    aggregation_module.set_init_model_para_dict(init_model_list)
    compress_module = CompressModule(compress_ratio=cur_arguments.compress_ratio,
                                     use_quan=cur_arguments.use_quan,
                                     use_para_decompose=cur_arguments.use_para_decom)

    # exit()
    #  init set matrix
    init_set_v1(client_list, cur_arguments, aggregation_module, compress_module)
    import time

    for epoch in range(1, cur_arguments.communication_round + 1):
        print("------" * 11)
        start = time.time()
        client_list = federate_train_v1(client_list, cur_arguments, epoch,
                                        aggregation_module=aggregation_module, compression_module=compress_module)
        end = time.time()
        train_time = end - start
        start = time.time()
        # print(epoch, ":finish_one_round:", 20 * '*', "result on train dataset")
        # federate_test(all_model_dict, new_federate_train_dataloader1, epoch, cur_arguments)
        print(epoch, ":finish_one_round:", 20 * '*', "result on test dataset")
        federate_test_v1(client_list, epoch)
        end = time.time()
        test_time = end - start

        start = time.time()
        #  save model checkpoint
        if epoch % 10 == 0:
            for index, cur_client in enumerate(client_list):
                current_model_local = cur_client.get_model()
                logger.save_client_model(current_model_local, index, "round_" + str(epoch))
            client_list = routine_set_v1(client_list, cur_arguments, aggregation_module, compress_module)
        end = time.time()
        routine_time = end - start
        print("Time cost is Train time:{:.6f},Test time:{:.6f},Routine time:{:.6f}".format(train_time, test_time,
                                                                                           routine_time))
        sys.stdout.flush()
    if re_stdout:
        sys.stdout = savedStdout  # 
        file.close()
