import torch
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from Algorithms.models.model import *

# num_inputs = 2
# num_examples = 20
# true_w = [2, -3.4]
# true_b = 4.2
# features = torch.randn(num_examples, num_inputs,
#                        dtype=torch.float32)
# labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
# labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
#                        dtype=torch.float32)

# def data_iter(batch_size, features, labels):
#     num_examples = len(features)
#     indices = list(range(num_examples))
#     random.shuffle(indices)  # 样本的读取顺序是随机的
#     for i in range(0, num_examples, batch_size):
#         j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
#         yield  features.index_select(0, j), labels.index_select(0, j)

# batch_size = 10

# for X, y in data_iter(batch_size, features, labels):
#     print(X, y)
#     break

# torch.manual_seed(0)

# flag = torch.rand(20)
# flag2 = torch.rand(20)
# users = range(20)

# selected = [users[index] for index, val in enumerate(flag) if val < 0.5]

# for index, val in enumerate(flag):
#     if val < 0.5:
#         print(index, val)

# print(selected)

# model = Net()
# model.cuda()
# while True:
#     model.train()
# alpha = torch.exp(torch.abs(list(model.parameters())[0].data))
# for index, val in enumerate(alpha):
#     sumCol = torch.sum(val)
#     alpha[index] = torch.div(val, sumCol.item())
# print(alpha)

# print(list(model.parameters())[0])
# for index, global_param in enumerate(model.parameters()):
#     if index == 0:
#         alpha = torch.div(features, sumFea.item())
#         print(alpha)
#         global_param.data = global_param.data.mul(alpha)
#         print(global_param.data)
# import os
# import torch.multiprocessing as mp
# import time
# torch.manual_seed(0)

# def init_server(name, q):
#     print(time.time())
#     print('Run child process %s (%s)...' % (name, os.getpid()))
#     model = Net()
#     # model.share_memory()
#     q.put(model)
#     print(time.time())
#     print("Queue size is ", q.qsize())
#     # x = torch.Tensor(10).cuda()
#     # q.put(x)
#     print("send a global model")
#     time.sleep(10)
#     # print(q.get())
#     # x = torch.Tensor(10)
#     # q.put(x)
# def init_client(name, q):
#     print(time.time())
#     print('Run child process %s (%s)...' % (name, os.getpid()))
#     model = q.get()
#     print(time.time())
#     print('Get a model, ', model)
#     print("Queue size is ", q.qsize())
#     del model

# def task(name,x, q):
#     print(time.time())
#     print(x)
#     print('Run child process %s (%s)...' % (name, os.getpid()))
#     time.sleep(1)
#     y = x*2
#     print(y)

# def server_task(x, q):
#     x = x*2
#     print('server',x)


# if __name__=='__main__':
#     mp.set_start_method('spawn')
#     print(torch.multiprocessing.get_all_sharing_strategies())
#     print('Parent process %s.' % os.getpid())
#     children = []
#     q = mp.Queue()
#     x = torch.Tensor(1)
#     x.share_memory_()
#     server = mp.Process(target=server_task, args=(x,q))
#     for i in range(3):
#         children.append(mp.Process(target=task, args=(i,x, q)))
#     print('Process will start.')
#     server.start()
#     for client in children:
#         client.start()
#     start_time = time.time()
#     for client in children:
#         client.join()
#     server.join()
#     end_time = time.time()
#     print('time: ', end_time - start_time)
#     print('Process end.')

# os.system('python3 server.py --dataset=MNIST --model=cnn --async_process=True --algorithm=FedAvg')
# os.system('python3 client.py --dataset=MNIST --async_process=True --batch_size=20 --learning_rate=0.008 --lamda=0.5 --beta=0.1 --num_global_iters=800 --optimizer=SGD --local_epochs=20 --algorithm=FedAvg --data_load=fixed --index=1')
model = Net()
model = model.cuda()
torch.save(model.state_dict(), os.path.join('models', 'test.pt'))
model_dict = torch.load(os.path.join('models', 'test.pt'))
model.load_state_dict(model_dict)
# print(model_dict)
# for param in model_dict.values():
#     print(param)
model.train()
index = 0
for old, new in zip(model.parameters(), model_dict.values()):
    old.data = new.data
    index = index + 1
    # print("old:",old)
    print("new:", new)
print(index)
print(model_dict.keys())