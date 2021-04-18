import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from Algorithms.optimizers.optimizer import ASOOptimizer
from Algorithms.users.userBase import User

class UserFedAsync(User):
    def __init__(self, id, train_data, test_data, model, async_process, batch_size, learning_rate, beta, lamda, local_epochs, optimizer, data_load, delay=0):
        super().__init__(id, train_data, test_data, model, async_process, batch_size, learning_rate, beta, lamda, local_epochs, optimizer, data_load)
        self.delay = delay
        self.delay_counter = 0
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.train_counter = 0
        self.check_flag = True
        self.sa = pow((self.delay+1),-0.5)*0.8
    
    def train(self, global_model):
        LOSS = 0
        # loss_log = []
        self.model.train()
        for p, new_param in zip(self.model.parameters(), global_model):
            p.data = new_param.data.clone()
        for epoch in range(1, self.local_epochs+1):
            self.model.train()
            X, y = self.get_next_train_batch()
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            # loss_log.append(loss.item())
            loss.backward()
            self.optimizer.step()
        self.trained = True
        # self.loss_log.append(loss_log)
        return LOSS

    def run(self, server):
        if self.check_flag is True:
            if self.trained is True:
                if self.delay_counter == self.delay:   
                    global_model = self.get_global_parameters(server) 
                    for local, glob in zip(self.model.parameters(), global_model):
                        local.data = local.data * self.sa + (1-self.sa)*glob.data
                    server.update_parameters(self.id, list(self.model.parameters()), self.train_data_samples)
                    self.delay_counter = 0
                    self.trained = False
                    self.check_flag = False
                    return
                else:
                    self.delay_counter = self.delay_counter + 1
                    return 
            # sync drop
            # if self.delay_counter < self.delay:
            #     self.delay_counter = self.delay_counter + 1
            #     return
            global_model = self.get_global_parameters(server)
            self.train(global_model)
            self.train_counter = self.train_counter + 1
            # sync drop
            # if self.delay == 0:
            #     server.update_parameters(self.id, list(self.model.parameters()), self.train_data_samples)
            # self.delay_counter = 0
            # self.check_flag = False

            if self.delay_counter == self.delay:
                server.update_parameters(self.id, list(self.model.parameters()), self.train_data_samples)
                self.check_flag = False
                self.delay_counter = 0
                self.trained = False
            else:
                self.delay_counter = self.delay_counter + 1
        else:
            global_model = self.get_global_parameters(server)
            self.train(global_model)
            self.train_counter = self.train_counter + 1
            self.check_flag = True
            server.update_parameters(self.id, list(self.model.parameters()), self.train_data_samples)
            self.trained = False