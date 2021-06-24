"""
time:20201216
"""
import os
import copy

import torch
import torch.nn.functional as F


class Client(object):

    def __init__(self,worker_id,train_dataset,test_dataset,model,*args,**kwargs):

        self.worker_id = worker_id
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = model

    def get_worker_id(self):
        return self.worker_id

    def get_train_dataset(self):
        return self.train_dataset

    def get_test_dataset(self):
        return self.test_dataset

    def get_model(self):
        return copy.deepcopy(self.model)

    def set_model(self,update_model):
        update_state_dict = update_model.state_dict()
        self.model.load_state_dict(update_state_dict)


def client_train_schedule(model,dataloader,optimizer,cur_arguments,device):
    """
    train schedule for one model proposed method
    :param model:
    :param dataloader:
    :param optimizer:
    :param cur_arguments:
    :param device:
    :return: model that after local update
    """
    local_epoch = cur_arguments.local_epoch
    for i in range(local_epoch):
        correct = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            current_model = model
            current_optimizer = optimizer
            for name, param in current_model.named_parameters():
                param.requires_grad = True
            current_optimizer.zero_grad()
            output = current_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            current_optimizer.step()
            if cur_arguments.local_two_update:    # MAML: 客户端上２次更新
                current_optimizer.zero_grad()
                output = current_model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                current_optimizer.step()

            pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
            correct = correct + pred.eq(target.view_as(pred)).sum().item()
    return model


def client_train_schedule_perfed(model,dataloader,optimizer,cur_arguments,device):
    """
    train schedule for one model perfed avg
    :param model:
    :param dataloader:
    :param optimizer:
    :param cur_arguments:
    :param device:
    :return: model that after local update
    """
    local_epoch = cur_arguments.local_epoch
    for i in range(local_epoch):
        correct = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            current_model = model
            current_optimizer = optimizer
            for name, param in current_model.named_parameters():
                param.requires_grad = True
            current_optimizer.zero_grad()
            output = current_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            current_optimizer.step()
            current_optimizer.zero_grad()
            output = current_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            current_optimizer.step()

            pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
            correct = correct + pred.eq(target.view_as(pred)).sum().item()
    return model


def client_train_schedule_fedavg(model,dataloader,optimizer,cur_arguments,device):
    """
    train schedule for one model fedavg
    :param model:
    :param dataloader:
    :param optimizer:
    :param cur_arguments:
    :param device:
    :return: model that after local update
    """
    local_epoch = cur_arguments.local_epoch
    for i in range(local_epoch):
        correct = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            current_model = model
            current_optimizer = optimizer
            for name, param in current_model.named_parameters():
                param.requires_grad = True
            current_optimizer.zero_grad()
            output = current_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            current_optimizer.step()
            pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
            correct = correct + pred.eq(target.view_as(pred)).sum().item()
    return model


def client_test_schedule(model,dataloader,optimizer,cur_arguments,device):
    pass




def client_train_schedule_step(model,dataloader,optimizer,cur_arguments,device,step):
    """
    train schedule for one model
    :param model:
    :param dataloader:
    :param optimizer:
    :param cur_arguments:
    :param device:
    :return: model that after local update
    """

    for batch_idx, (data, target) in enumerate(dataloader):
        if batch_idx >= step:
            break
        data, target = data.to(device), target.to(device)
        current_model = model
        current_optimizer = optimizer
        for name, param in current_model.named_parameters():
            param.requires_grad = True
        current_optimizer.zero_grad()
        output = current_model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        current_optimizer.step()
    return model



