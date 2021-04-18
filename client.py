import zerorpc
import os
import argparse
import torch
import time
import gevent
import random
import numpy as np
from Algorithms.users.userASO import UserASO
from Algorithms.users.userFedAvg import UserFedAvg
from Algorithms.users.userFAFed import UserFAFed
from utils.model_utils import read_user_data_async
from Algorithms.models.model import *
import pandas as pd
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic =True
torch.backends.cudnn.benchmark = False
def init_client(dataset, model, algorithm, async_process, batch_size, learning_rate, lamda, beta, num_glob_iters, local_epochs, optimizer, data_load, index):
    rpcController = zerorpc.Client()
    rpcController.connect('tcp://127.0.0.1:8888')
    start_time = time.time()
    id, train, test = read_user_data_async(index, dataset)
    rpcController.add_client(id, 0)
    if(model == "mclr"):
        if(dataset == "Cifar10"):
            model = Mclr_Logistic(60,10)
        else:
            model = Mclr_Logistic()
    if(model == "cnn"):
        if(dataset == "Cifar10"):
            model = CifarNet()
        else:
            model = Net()
    model = model.cuda()
    pre_model = torch.load(os.path.join('models', 'server_'+id+'.pt'))
    model.load_state_dict(pre_model)
    if algorithm == 'FedAvg':
        user = UserFedAvg(id, train, test, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load)
    if algorithm == 'ASO':
        user = UserASO(id, train, test, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load)
    if algorithm == 'FAFed':
        user = UserFAFed(id, train, test, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load)
    print(user.id, ' start !')
    user.save_model()


    for i in range(num_glob_iters):
        if user.can_train() is False:
            continue
        else:
            if user.check_async_update():
                rpcController.client_update(user.id, user.train_data_samples)
                user.trained = False
        
        time.sleep(torch.randint(10, 100, (1,)).item() / 1000)   
        rpcController.get_model(user.id)         
        global_model = user.load_model('server_'+user.id)
        user.train(global_model)
        user.save_model()
        
        if user.check_async_update():
            rpcController.client_update(user.id, user.train_data_samples)
            user.trained = False
        if i % 10 == 0:
            user.test()
            print(user.id, ' test_acc is', user.test_acc, ' in ', i)
        else:
            print(user.id, ' in ', i)

    rpcController.close_client(user.id)
    print(user.id, ' is over.')
    user.test()
    dictData = {}
    dictData[user.id+'_test_acc'] = user.test_acc_log[:]
    dataFrame = pd.DataFrame(dictData)
    filename = './results/'+algorithm+'_'+dataset+user.id+'_'+'.csv'
    dataFrame.to_csv(filename, index=False, sep=',')
    end_time = time.time()
    print(user.id, " solve time si", end_time - start_time)

class ClientSub(object):
    def __init__(self, dataset, model, algorithm, async_process, batch_size, learning_rate, lamda, beta, num_glob_iters, local_epochs, optimizer, data_load, index):
        self.dataset = dataset
        self.algorithm = algorithm
        self.clientRpc = zerorpc.Client()
        self.clientRpc.connect('tcp://127.0.0.1:8888')
        self.start_time = time.time()
        id, train, test = read_user_data_async(index, dataset)
        if(model == "mclr"):
            if(dataset == "Cifar10"):
                model = Mclr_Logistic(60,10)
            else:
                model = Mclr_Logistic()
        if(model == "cnn"):
            if(dataset == "Cifar10"):
                model = CifarNet()
            else:
                model = Net()
        model = model.cuda()
        pre_model = torch.load(os.path.join('models', 'server.pt'))
        model.load_state_dict(pre_model)
        if algorithm == 'FedAvg':
            user = UserFedAvg(id, train, test, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load)
        if algorithm == 'ASO':
            user = UserASO(id, train, test, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load)
        if algorithm == 'FAFed':
            user = UserFAFed(id, train, test, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load)
        print(user.id, ' start !')
        user.save_model()
        self.clientRpc.add_client(id, user.train_data_samples)
        self.user = user
    
    def new_epoch(self, epoch ,time_stamp):
        print(self.user.id, ' get ', epoch)
        if self.user.can_train() is False:
            print(self.user.id, 'pass')
            self.clientRpc.close_epoch(self.user.id, False)
            return False
        print(self.user.id, 'train')
        time.sleep(torch.randint(10, 100, (1,)).item() / 1000)   
        self.clientRpc.get_model(self.user.id)         
        global_model = self.user.load_model('server_'+self.user.id)
        self.user.train(global_model)
        self.user.save_model()
        self.clientRpc.client_update(self.user.id, self.user.train_data_samples)
        self.clientRpc.close_epoch(self.user.id, True)

    def ready_start(self, time_stamp):
        print(self.user.id, ' ready start.', time_stamp)

    def complete_train(self, time_stamp):
        print(self.user.id, ' is over.')
        self.user.test()
        dictData = {}
        dictData[self.user.id+'_test_acc'] = self.user.test_acc_log[:]
        dataFrame = pd.DataFrame(dictData)
        filename = './results/'+self.algorithm+'_'+self.dataset+self.user.id+'_'+'.csv'
        dataFrame.to_csv(filename, index=False, sep=',')
        end_time = time.time()
        print(self.user.id, " solve time is", end_time - self.start_time)
        self.clientRpc.close_client(self.user.id)
        self.clientRpc.close()

def main(dataset, model, algorithm, async_process, batch_size, learning_rate, lamda, beta, num_glob_iters, local_epochs, optimizer, data_load, index):
    if async_process is True:
        init_client(dataset, model, algorithm, async_process, batch_size, learning_rate, lamda, beta, num_glob_iters, local_epochs, optimizer, data_load, index)
    else:
        client = zerorpc.Subscriber(ClientSub(dataset, model, algorithm, async_process, batch_size, learning_rate, lamda, beta, num_glob_iters, local_epochs, optimizer, data_load, index))
        client.connect('tcp://127.0.0.1:8889')
        client.run()

        
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
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--algorithm", type=str, default="FedAvg",choices=["FedAvg", "ASO", "FAFed"]) 
    parser.add_argument("--data_load", type=str, default="fixed", choices=["fixed", "flow"], help="user data load")
    parser.add_argument("--index", type=int)
    args = parser.parse_args()

    main(
        dataset=args.dataset,
        model=args.model,
        algorithm = args.algorithm,
        async_process=args.async_process,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lamda = args.lamda,
        beta = args.beta,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        optimizer=args.optimizer,
        data_load = args.data_load,
        index = args.index
        )