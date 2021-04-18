import torch
import os
import copy
from torch.utils.data import DataLoader
from utils.model_utils import Object
class Server:
    def __init__(self, algorithm, model, async_process, test_data, batch_size):
        self.model = copy.deepcopy(model)
        self.algorithm = algorithm
        self.batch_size = batch_size
        self.test_data = test_data
        self.async_process = async_process
        self.testloader = DataLoader(test_data, batch_size)
        self.test_acc = 0
        self.test_acc_log = []

        self.status = False
        self.update_list = []
        self.users = {}

    def append_user(self, id, samples, delay = 0):
        self.users[id] = Object(dict(id=id, model=copy.deepcopy(list(self.model.parameters())), samples=samples, delay=delay))

    def update_parameters(self, id, new_parameters, sample_len, delay = 0):
        self.append_update_cache(id, new_parameters, sample_len, delay)
        if self.async_process == True and len(self.update_list) != 0:
            self.clear_update_cache()
    
    def append_update_cache(self, id, new_parameters, samples_len, delay = 0):
        if id in self.users:
            self.update_list.append(Object(dict(id=id, model=new_parameters, samples=samples_len, delay=delay)))
        else:
            self.users[id] = Object(dict(id=id, model=new_parameters,samples=samples_len, delay=delay))
            print('append user, ', id)

    def clear_update_cache(self):
        cache = self.update_list[:]
        self.update_list = []
        self.status = True
        self.aggregate_parameters(cache)
        # for user_data in cache:
        #     self.aggregate_parameters(user_data)
        self.status = False
        if len(self.update_list) != 0:
            print("clear cache again")
            self.clear_update_cache()
    
    def test(self):
        self.model.eval()
        test_acc = 0
        for i, (x, y) in enumerate(self.testloader):
            output = self.model(x.cuda())
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y.cuda())).item()
        self.test_acc = test_acc*1.0 / len(self.test_data)
        self.test_acc_log.append(self.test_acc)
        return test_acc, len(self.test_data)
    
    def save_model(self, name="server"):
        model_path = os.path.join("models")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model.state_dict(), os.path.join(model_path, name+'.pt'))
        # torch.save(self.model, os.path.join(model_path, "server" + ".pt"))

    def load_model(self, name="server"):
        model_path = os.path.join("models")
        assert (os.path.exists(model_path))
        model = torch.load(os.path.join("models", name+".pt"))
        return model.values()

    def model_exists(self, name):
        return os.path.exists(os.path.join("models", name + ".pt"))