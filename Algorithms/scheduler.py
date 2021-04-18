import torch
import os
import json
import numpy as np
import copy
from Algorithms.servers.serverASO import ServerASO
from Algorithms.servers.serverFedAvg import ServerFedAvg
from Algorithms.servers.serverLGP import ServerLGP
from Algorithms.servers.serverPerFed import ServerPerFed
from Algorithms.servers.serverFedAsync import serverFedAsync
from Algorithms.users.userASO import UserASO
from Algorithms.users.userFedAvgBase import UserFedAvg
from Algorithms.users.userLGP import UserLGP
from Algorithms.users.userPerFed import UserPerFed
from Algorithms.users.userFedAsync import UserFedAsync
from utils.model_utils import read_data, read_user_data
import torch
import pandas as pd

class Scheduler:
    def __init__(self, dataset,algorithm, model, async_process, batch_size, learning_rate, lamda, beta, num_glob_iters,
                 local_epochs, optimizer, num_users, user_labels, niid, times, data_load, extra):
        self.dataset = dataset
        self.model = copy.deepcopy(model)
        self.algorithm = algorithm
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.async_process = async_process
        self.lamda = lamda
        self.beta = beta
        self.times = 8
        self.data_load = data_load
        self.extra = extra
        self.num_users = num_users
        self.num_glob_iters = num_glob_iters
        self.local_epochs = local_epochs
        self.user_labels = user_labels
        self.niid = niid
        self.users = []
        self.local_acc = []
        self.avg_local_acc = []
        self.avg_local_train_acc = []
        self.avg_local_train_loss = []
        self.server_acc = []
        # old data split
        data = read_data(dataset, niid, num_users, user_labels)
        self.num_users = num_users
        test_data = []
        # id, train, test = read_user_data(0, data, dataset)
        # if algorithm == 'FedAvg':
        #     user = UserFedAvg(id, train, test, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load, self.times)
        # if algorithm == 'ASO':
        #     user = UserASO(id, train, test, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load, self.times)
        # if algorithm == 'LGP':
        #     user = UserLGP(id, train, test, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load, self.times)
        # if algorithm == 'PerFed':
        #     user = UserPerFed(id, train, test, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load, self.times)
        # self.users.append(user)
        # test_data.extend(test)
        for i in range(self.times):
            id, train, test = read_user_data(i, data, dataset)
            if algorithm == 'FedAvg':
                user = UserFedAvg(id, train, test, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load, i+1)
            if algorithm == 'ASO':
                user = UserASO(id, train, test, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load, i+1)
            if algorithm == 'LGP':
                user = UserLGP(id, train, test, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load, i+1)
            if algorithm == 'PerFed':
                user = UserPerFed(id, train, test, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load, i+1)
            if algorithm == 'FedAsync':
                user = UserFedAsync(id, train, test, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load, i+1)
            self.users.append(user)
            test_data.extend(test)
        for i in range(self.times, self.num_users):
        # for i in range(self.num_users):
            id, train, test = read_user_data(i, data, dataset)
            if algorithm == 'PerFed':
                user = UserPerFed(id, train, test, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load)
            if algorithm == 'FedAvg':
                user = UserFedAvg(id, train, test, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load)
            if algorithm == 'ASO':
                user = UserASO(id, train, test, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load)
            if algorithm == 'LGP':
                user = UserLGP(id, train, test, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load)
            if algorithm == 'FedAsync':
                user = UserFedAsync(id, train, test, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load)
            self.users.append(user)
            test_data.extend(test)
        if algorithm == 'FedAvg':
            self.server = ServerFedAvg(algorithm, model, async_process, test_data, batch_size)
        if algorithm == 'PerFed':
            self.server = ServerPerFed(algorithm, model, async_process, test_data, batch_size)
        if algorithm == 'ASO':
            self.server = ServerASO(algorithm, model, async_process, test_data, batch_size)
        if algorithm == 'LGP':
            self.server = ServerLGP(algorithm, model, async_process, test_data, batch_size)
        if algorithm == 'FedAsync':
            self.server = serverFedAsync(algorithm, model, async_process, test_data, batch_size)
        for user in self.users:
            self.server.append_user(user.id, user.train_data_samples)
    
    def run(self):
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            for user in self.users:
                user.run(self.server)
            if self.async_process == False:
                self.server.clear_update_cache()
            self.evaluate()
        # sync not drop
        # extra_iters = [800,800, 400, 267, 200, 160, 134,115, 100, 89]
        # for i in range(extra_iters[self.times] - self.users[0].train_counter):
        #     user = self.users[0]
        #     user.train(list(self.server.model.parameters()))
        #     self.server.update_parameters(user.id, user.model.parameters(), user.train_data_samples)
        #     self.server.clear_update_cache()
        #     self.evaluate()
        
        # async
        # train_count = []
        # for user in self.users:
        #     if user.trained:
        #         self.server.update_parameters(user.id, user.model.parameters(), user.train_data_samples)
        #     train_count.append(user.train_counter)
        # self.server.clear_update_cache()
        # self.evaluate()
        # self.local_acc.append(train_count)
        # self.server_acc.append(self.num_glob_iters)
        
        self.save_results()
        self.server.save_model()
        # self.save_loss_log()
    
    def save_loss_log(self):
        for user in self.users:
            loss_log = user.loss_log
            name=range(21)
            dataframe = pd.DataFrame(columns=name, data=loss_log)
            fileName = "./logs/"+user.id+'.csv'
            dataframe.to_csv(fileName, index=False, sep=',')
    
    def evaluate(self):
        self.evaluate_users()
        self.evaluate_server()

    def evaluate_users(self):
        stats = self.users_test() 
        client_acc = [x*1.0/y for x, y in zip(stats[2], stats[1])] 
        self.local_acc.append(client_acc)
        print("Local Accurancy: ", client_acc)

    def evaluate_server(self):
        stats = self.server.test()
        server_acc = stats[0]*1.0/stats[1]
        self.server_acc.append(server_acc)
        print("Central Model Accurancy: ", server_acc)
    
    def users_test(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct

    def users_train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss() 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct, losses
    
    def save_results(self):
        alg = self.dataset + "_" + self.algorithm + "_" + self.optimizer
        if self.async_process == True:
            alg = alg + "_async"
        else:
            alg = alg + "_sync"
        if self.niid == True:
            alg = alg + "_niid"
        else:
            alg = alg + "_iid"
        alg = alg + "_" + str(self.learning_rate) + "_" + str(self.beta) + "_" + str(self.lamda) + "_" + str(self.num_users) + "u" + "_" + str(self.user_labels) + "l" + "_" + str(self.batch_size) + "b" + "_" + str(self.local_epochs) + "_" + str(self.num_glob_iters) + "ep" + "_" + self.data_load
        alg = alg + "_" + str(self.times) + "_" + self.extra
        if (len(self.server_acc) &  len(self.local_acc) ) :
            dictData={}
            for i in range(self.num_users):
                dictData['client_'+str(i)] = [x[i] for x in self.local_acc]
            dictData['central_model_acc'] = self.server_acc[:]
            dataframe = pd.DataFrame(dictData)
            fileName = "./results/"+alg+'.csv'
            dataframe.to_csv(fileName, index=False, sep=',')
        
    

    
        




         
