import torch
import os
import copy
from Algorithms.servers.serverBase import Server

class ServerPerFed(Server):
    def __init__(self, algorithm, model, async_process, test_data, batch_size):
        super().__init__(algorithm, model, async_process, test_data, batch_size)
    
    def aggregate_parameters(self, user_data):
        if self.async_process == True:
            for user_updated in user_data:  
                self.users[user_updated.id].samples = user_updated.samples
                total_train = 0
                for user in self.users.values():
                    total_train += user.samples
                for global_param, user_old_param, user_new_param in zip(self.model.parameters(), self.users[user_updated.id].model, user_updated.model):
                    # global_param.data = global_param.data - (user_updated.samples / total_train)*(user_old_param.data - user_new_param.data)
                    global_param.data = global_param.data - (1 / len(self.users))*(user_old_param.data - user_new_param.data)
        else:
            for user_updated in user_data:
                self.users[user_updated.id].samples = user_updated.samples
                for new_param, old_param in zip(user_updated.model, self.users[user_updated.id].model):
                    old_param.data = new_param.data.clone()
            total_train = 0
            for user in self.users.values():
                total_train += user.samples
            for index, global_copy in enumerate(self.model.parameters()):
                global_copy.data = torch.zeros_like(global_copy.data)
                for user in self.users.values():
                    global_copy.data = global_copy.data + (user.samples / total_train)*user.model[index].data