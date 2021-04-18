import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from Algorithms.optimizers.optimizer import ASOOptimizer
from Algorithms.users.userBase import User

class UserFedAvg(User):
    def __init__(self, id, train_data, test_data, model, async_process, batch_size, learning_rate, beta, lamda, local_epochs, optimizer, data_load):
        super().__init__(id, train_data, test_data, model, async_process, batch_size, learning_rate, beta, lamda, local_epochs, optimizer, data_load)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
    
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
        if self.can_train() == False:
            return False
        else: 
            if self.check_async_update():
                server.update_parameters(self.id, self.model.parameters(), self.train_data_samples)
                self.trained = False
        global_model = self.get_global_parameters(server)
        self.train(global_model)

        if self.check_async_update():
            server.update_parameters(self.id, self.model.parameters(), self.train_data_samples)
            self.trained = False
