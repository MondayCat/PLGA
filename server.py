import zerorpc
import os
import argparse
import torch
import time
import random
import numpy as np
import gevent
from Algorithms.servers.serverASO import ServerASO
from Algorithms.servers.serverFedAvg import ServerFedAvg
from Algorithms.servers.serverFAFed import ServerFAFed
from utils.model_utils import read_test_data_async
from Algorithms.models.model import *
import pandas as pd
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic =True
torch.backends.cudnn.benchmark = False
trigger = gevent.event.Event()
class RpcController(object):
    def __init__(self, dataset, algorithm, model, num_users, async_process, batch_size):
        self.dataset = dataset
        self.num_users = num_users
        test_data = read_test_data_async(dataset)
        if(model == "mclr"):
            if(dataset == "Cifar10"):
                pre_model = Mclr_Logistic(60,10)
            else:
                pre_model = Mclr_Logistic()
        if(model == "cnn"):
            if(dataset == "Cifar10"):
                pre_model = CifarNet()
            else:
                pre_model = Net()
        pre_model = pre_model.cuda()
        pre_model.eval()
        if algorithm == 'FedAvg':
            server = ServerFedAvg(algorithm, pre_model, async_process, test_data, batch_size)
        if algorithm == 'ASO':
            server = ServerASO(algorithm, pre_model, async_process, test_data, batch_size)
        if algorithm == 'FAFed':
            server = ServerFAFed(algorithm, pre_model, async_process, test_data, batch_size)
        # server.test()
        self.start_time = time.time()
        self.server = server
        self.nun_users = num_users
        self.server.save_model()
        self.client_list = {}
        self.client_counter = 0
        self.aggerate_counter = 0
        self.user_epoch_counter = 0
        # self.read = {}
        # self.write = False
        # self.start_train = False
    def client_update(self, id, samples_len):
        try:
            # if self.start_train is False:
            #     print('not start')
            #     return False
            userModel = self.server.load_model("client_"+id)
            self.server.update_parameters(id, userModel, samples_len)

            self.server.save_model()
            self.aggerate_counter = self.aggerate_counter + 1
            if self.aggerate_counter % 10 == 0:
                self.server.test()
                print('Server updated ', id, ', test_acc is ', self.server.test_acc)
            else:
                print('Server updated ', id)
            return True
        except Exception as e:
            print('client update error', e)
            return False

    def add_client(self, id, samples):
        try:
            # self.read[id] = False
            self.server.save_model("server_"+id)
            self.server.append_user(id, samples)
            self.client_counter = self.client_counter + 1
            print('Append Client, ',id)
            if self.client_counter == self.num_users:
                trigger.set()
                print("Start Training !")
            return True
        except Exception as e:
            print('add client error', e)
            return False

    def get_model(self, id):
        if self.server.algorithm == 'ASO':
            self.server.save_model("server_"+id)
            return True
        for global_param, user_init in zip(self.server.model.parameters(), self.server.users[id].model):
            user_init.data = global_param.data.clone()
        self.server.save_model("server_"+id)
        return True

    def close_client(self, id):
        self.client_counter = self.client_counter - 1
        if self.client_counter == 0:
            self.server.test()
            dictData = {}
            dictData['server_test_acc'] = self.server.test_acc_log[:]
            dataFrame = pd.DataFrame(dictData)
            filename = './results/'+self.server.algorithm+'_'+self.dataset+'_'+'.csv'
            dataFrame.to_csv(filename, index=False, sep=',')
            print('Server Over !')
            end_time = time.time()
            print("Solve time is, ", end_time - self.start_time)
            trigger.set()
    
    def close_epoch(self, id, isTrain):
        self.user_epoch_counter = self.user_epoch_counter + 1
        print(id, "close epoch, ", isTrain, self.user_epoch_counter, self.client_counter, self.num_users)
        if self.user_epoch_counter == self.client_counter:
            self.server.clear_update_cache()
            self.user_epoch_counter = 0
            trigger.set()


def main(dataset, algorithm, model, num_users, async_process, batch_size, num_global_iters):
    if async_process is True:
        print("async structure")
        server = zerorpc.Server(RpcController(dataset, algorithm, model, num_users, async_process, batch_size))
        server.bind('tcp://0.0.0.0:8888')
        print('Server Start!')
        server.run()
    else:
        print("sync structure")
        server = zerorpc.Server(RpcController(dataset, algorithm, model, num_users, async_process, batch_size))
        server.bind('tcp://0.0.0.0:8888')
        publisher = zerorpc.Publisher()
        publisher.bind('tcp://0.0.0.0:8889')
        print('Server Start!')
        trigger.clear()
        gevent.spawn(server.run)
        trigger.wait()
        publisher.ready_start(time.time())
        time.sleep(1)
        for i in range(num_global_iters):
            print("Epoch ", i)
            trigger.clear()
            publisher.new_epoch(i, time.time())
            trigger.wait()
        trigger.clear()
        publisher.complete_train(time.time())
        trigger.wait()
        
    
    

         
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):

        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "FashionMNIST", "Cifar10"])
    parser.add_argument("--model", type=str, default="cnn", choices=["mclr", "cnn"])
    parser.add_argument("--async_process", type=str2bool, default=True)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Local learning rate")
    parser.add_argument("--lamda", type=float, default=1.0, help="Regularization term")
    parser.add_argument("--beta", type=float, default=0.001, help="Decay Coefficient")
    parser.add_argument("--num_global_iters", type=int, default=800)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="FedAvg",choices=["FedAvg", "ASO", "FAFed"]) 
    parser.add_argument("--numusers", type=int, default=10, help="Number of Users per round")
    parser.add_argument("--user_labels", type=int, default=5, help="Number of Labels per client")
    parser.add_argument("--niid", type=str2bool, default=True, help="data distrabution for iid or niid")
    parser.add_argument("--times", type=int, default=5, help="running time")
    parser.add_argument("--data_load", type=str, default="fixed", choices=["fixed", "flow"], help="user data load")
    args = parser.parse_args()

    main(dataset=args.dataset, algorithm=args.algorithm, model=args.model, num_users=args.numusers, async_process=args.async_process, num_global_iters=args.num_global_iters, batch_size=args.batch_size)

