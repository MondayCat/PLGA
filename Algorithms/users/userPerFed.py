import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
import json
from torch.utils.data import DataLoader
from Algorithms.optimizers.optimizer import MySGD
from Algorithms.users.userBase import User

class UserPerFed(User):
    def __init__(self, id, train_data, test_data, model, async_process, batch_size, learning_rate, beta, lamda, local_epochs, optimizer, data_load, delay=0):
        super().__init__(id, train_data, test_data, model, async_process, batch_size, learning_rate, beta, lamda, local_epochs, optimizer, data_load)
        self.delay = delay
        self.delay_counter = 0
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = MySGD(self.model.parameters(), lr=learning_rate)
        self.last_model = copy.deepcopy(list(model.parameters()))
        self.train_counter = 0
    
    def train(self, global_model):
        LOSS = 0
        # loss_log = []
        self.model.train()
        for p, new_param in zip(self.model.parameters(), global_model):
            p.data = new_param.data.clone()
        for epoch in range(1, self.local_epochs+1):
            for p, last in zip(self.model.parameters(), self.last_model):
                last.data = p.data.clone()
            self.model.train()
            X, y = self.get_next_train_batch()
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            # loss_log.append(loss.item())
            loss.backward()
            self.optimizer.step()

            # X, y = self.get_next_train_batch()
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step(beta = self.beta)
        self.trained = True
        # self.loss_log.append(loss_log)
        return LOSS

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