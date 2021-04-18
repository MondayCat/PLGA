from torch.optim import Optimizer
import torch
import copy
class ASOOptimizer(Optimizer):
    def __init__(self, params, lr, lamda, beta, mu = 0.001):
        defaults = dict(lr=lr, lamda=lamda, beta=beta)
        super(ASOOptimizer, self).__init__(params.parameters(), defaults)

        self.sk_grad = copy.deepcopy(list(params.parameters()))
        self.hk = copy.deepcopy(list(params.parameters()))
        for param, sk, hk in zip(params.parameters(), self.sk_grad, self.hk):
            sk.data = torch.zeros_like(param.data)
            hk.data = torch.zeros_like(param.data)
    # no dynamic_step_size  
    def step(self, local_weight=None):
        loss = None
        for group in self.param_groups:
            for p, local, pre_sk_grad, pre_hk in zip( group['params'], local_weight, self.sk_grad, self.hk):
                p.data = p.data - group['lr']*(p.grad.data + group['lamda']*(p.data - local))

        return loss

class pFedMeOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1 , mu = 0.001):
        #self.local_weight_updated = local_weight # w_i,K
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lamda=lamda, mu = mu)
        super(pFedMeOptimizer, self).__init__(params, defaults)
    
    def step(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip( group['params'], weight_update):
                p.data = p.data - group['lr'] * (p.grad.data + group['lamda'] * (p.data - localweight.data) + group['mu']*p.data)
        return  group['params'], loss
    
    def update_param(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip( group['params'], weight_update):
                p.data = localweight.data
        #return  p.data
        return  group['params']

class MySGD(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(MySGD, self).__init__(params, defaults)

    def step(self, beta = 0, last_model=None):
        loss = None
        for group in self.param_groups:
            # print(group)
            if last_model is None:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    if(beta != 0):
                        p.data.add_(-beta, d_p)
                        
                    else:     
                        p.data.add_(-group['lr'], d_p)
            else:
                for p, last in zip(group['params'], last_model):
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    p.data = last.data - beta*d_p
        return loss