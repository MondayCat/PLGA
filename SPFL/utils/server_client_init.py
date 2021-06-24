import torch
import syft as sy


def create_server_client(client_num):
    """
    create server and client worker
    :param client_num:
    :return: :a list  contain [server client1 client2]
    """
    # server_str = 'server'
    client_str = 'client'
    workers_list = []
    hook = sy.TorchHook(torch)
    # workers_list.append(sy.VirtualWorker(hook, id=server_str))
    for i in range(client_num):
        workers_list.append(sy.VirtualWorker(hook, id=client_str+str(i)))
    return workers_list
