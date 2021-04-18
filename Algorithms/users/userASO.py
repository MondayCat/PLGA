import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import copy
from torch.utils.data import DataLoader
from Algorithms.optimizers.optimizer import ASOOptimizer
from Algorithms.users.userBase import User

class UserASO(User):
    def __init__(self, id, train_data, test_data, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load, delay=0):
        super().__init__(id, train_data, test_data, model, async_process, batch_size, learning_rate, lamda, beta, local_epochs, optimizer, data_load)
        self.delay = delay
        self.delay_counter = 0
        self.train_counter = 0
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = ASOOptimizer(self.model, lr=self.learning_rate, lamda=self.lamda, beta=self.beta)
        self.last_model = copy.deepcopy(list(model.parameters()))
    
    def run(self, server):
        if self.trained is True:
            if self.delay_counter == self.delay:
                server.update_parameters(self.id, self.model.parameters(), self.train_data_samples)
                self.delay_counter = 0
                self.trained = False
                # sync not drop
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
        # server.update_parameters(self.id, self.model.parameters(), self.train_data_samples)
        # self.delay_counter = 0

        if self.delay_counter == self.delay:
            server.update_parameters(self.id, self.model.parameters(), self.train_data_samples)
            self.delay_counter = 0
            self.trained = False
        else:
            self.delay_counter = self.delay_counter + 1

    def train(self, global_model):
        self.model.train()
        # loss_log = []
        for p, new_param in zip(self.model.parameters(), global_model):
            p.data = new_param.data.clone()
        for epoch in range(1, self.local_epochs+1):
            self.model.train()
            X, y = self.get_next_train_batch()
            for i in range(30):
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step(self.last_model)
            for param, local in zip(self.model.parameters(), self.last_model):
                local.data = local.data - self.lamda*self.learning_rate*(local.data - param.data)
        for param, local in zip(self.model.parameters(), self.last_model):
            param.data = local.data.clone()
            
        self.trained = True
        # self.loss_log.append(loss_log)


