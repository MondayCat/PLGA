"""
add a weight matrix for proposed method v1
update schedule based on v2
根据第一步的参数决定不同客户端的分布以及相关矩阵的计算
copy from v4, use for model compress ,
model para sended is grad parameters
not original model parameters

ime:20210221
"""

import copy
import sys
sys.path.append("..")

import torch

from arguments_v1 import ExperimentPara
from compress_utils.compress_v3 import CompressModule,compress_fun


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class AggregationModule:
    """
    多个客户端权重融合的算法
    第四个版本
    new update schedule base
    """

    def __init__(self, cur_arguments: ExperimentPara,init_model):

        self.cur_arguments = cur_arguments
        self.cur_model_list = None
        self.last_step_model_list = None  # 保存上次迭代模型的参数,字典类型

        self.client_num = self.cur_arguments.client_num
        self.init_model = init_model

        self.base_para_matrix = torch.zeros((self.client_num, self.client_num))
        self.head_para_matrix = torch.zeros((self.client_num, self.client_num))
        self.base_para_matrix_last = torch.zeros((self.client_num, self.client_num))
        self.head_para_matrix_last = torch.zeros((self.client_num, self.client_num))

        self.last_state_dict_list = None

        self.compress_module = CompressModule(use_quan=False,
                                              use_para_decompose=False)

        self.local_stage_index = 0
        self.local_round_num = 10
        self.cos_max_val = 2

    def set_init_model_para_dict(self, init_model_list):
        """
        set init model para dict
        :param init_model_list:
        :return:
        """
        # 得到每个客户端的model的参数(weight,bias等)
        self.last_state_dict_list = [copy.deepcopy(init_model_list[i].state_dict())
                                     for i in range(self.client_num)]

    def set_init_model(self, init_model):
        """
        set init model
        for weight analysing between different client
        :return:
        """
        self.init_model = init_model

    def set_init_weight_matrix(self):
        """
        set init weight matrix
        :return:
        """
        x = torch.zeros((self.client_num, self.client_num))
        self.head_para_matrix = torch.zeros((self.client_num, self.client_num))
        self.base_para_matrix_last = torch.zeros((self.client_num, self.client_num))
        self.head_para_matrix_last = torch.zeros((self.client_num, self.client_num))

    def get_last_weight_matrix(self):
        print("I am doing it")
        print(self.base_para_matrix_last,self.head_para_matrix_last)
        return self.base_para_matrix_last,self.head_para_matrix_last

    def set_last_weight_matrix(self,base_para_matrix_last,head_para_matrix_last):
        print("I set it")
        self.base_para_matrix_last = base_para_matrix_last
        self.head_para_matrix_last = head_para_matrix_last

    def update_weight_matrix(self, update_model_list):
        """
        更新权重矩阵
        :param update_model_list:最新更新的模型
        :return:
        """
        i = 0
        cur_state_dict_list = [copy.deepcopy(update_model_list[i].state_dict())
                               for i in range(self.client_num)]

        init_model_dict = self.init_model.state_dict()

        base_para_matrix = torch.zeros((self.client_num,self.client_num))
        head_para_matrix = torch.zeros((self.client_num,self.client_num))
        base_para_matrix_last = torch.zeros((self.client_num,self.client_num))
        base_para_matrix_final = torch.zeros((self.client_num,self.client_num))
        head_para_matrix_last = torch.zeros((self.client_num,self.client_num))
        head_para_matrix_final = torch.zeros((self.client_num, self.client_num))

        from collections import defaultdict
        divergence_dict = dict()
        for key_name in init_model_dict.keys():  ##  init_model_dict: 'conv1.weight': [], 'conv2.weight': [], 'fc1.weight', 'fc2.weight'
            # print("key_name:",key_name)      ##conv1.weight   2个客户端
            share_para = init_model_dict[key_name]
            # print("share_para:", init_model_dict[key_name].size())  #[20,3,5,5]  [20]  [50,20,5,5] [50] [500,1250] [500]  [10,500] [10]
            diver_matrix = torch.zeros((self.client_num, self.client_num))
            for model_index in range(self.client_num):
                cur_param_dict = cur_state_dict_list[model_index]
                # print("cur_param_dict", cur_param_dict)
                cur_para = cur_param_dict[key_name]     #第i个客户端更新后的参数
                print("cur_para_size:", cur_para.size())
                print("cur_para", cur_para)
                for t_model_index in range(self.client_num):
                    tmp_param_dict = cur_state_dict_list[t_model_index]
                    tmp_para = tmp_param_dict[key_name]      #第j个客户端更新后的参数
                    diver_matrix[model_index, t_model_index] = torch.mean(
                        (cur_para - share_para) * (tmp_para - share_para)) \
                                                               / (torch.sqrt(
                        torch.mean((cur_para - share_para) ** 2)) * torch.sqrt(
                        torch.mean((tmp_para - share_para) ** 2)))
            divergence_dict[key_name] = diver_matrix     #每一层生成一个权重矩阵
            # print("divergence_dict",divergence_dict.keys(),divergence_dict.values())
        base_para_layer_count = 0
        head_para_layer_count = 0
        layer_weight_dict = {'conv1.weight': 0.5,
                             'conv2.weight': 0.3,
                             'fc1.weight': 0.2,
                             }
        for key_name,matrix in divergence_dict.items():

            if 'bias' in key_name:
                continue
            # print(key_name)
            # print(matrix)
            if 'fc2' in key_name:
                head_para_layer_count+=1
                head_para_matrix += matrix
            else:
                base_para_layer_count+=1
                base_para_matrix+=matrix*layer_weight_dict[key_name]  #前三层按权重加起来得到一个总的矩阵

        # self.base_para_matrix = base_para_matrix*2
        # self.head_para_matrix = head_para_matrix*6

        # compute last matrix

        base_para_matrix_last,head_para_matrix_last = self.get_last_weight_matrix()     #get last weight matrix
        head_para_matrix_final = a * head_para_matrix_last + (1 - a) * head_para_matrix    # add before matrix head

        print("base_para_matrix_last", base_para_matrix_last)
        print("head_para_matrix_last", head_para_matrix_last)
        print("base para matrix", base_para_matrix)
        print("head para matrix", head_para_matrix)

        base_para_matrix_final = a * base_para_matrix_last + (1 - a) * base_para_matrix   # add before matrix base

        # self.head_para_matrix_last = self.set_last_weight_matrix(base_para_matrix, head_para_matrix)

        self.base_para_matrix = base_para_matrix_final
        self.head_para_matrix = head_para_matrix_final
        print("head_para_matrix_final",head_para_matrix_final)
        print("base_para_matrix_final",base_para_matrix_final)


    def update_weight_matrix_one(self, update_model_list): # 客户端训练后的数据更新
        """
        更新权重矩阵,use only one matrix
        :param update_model_list:最新更新的模型
        :return:
        """
        cur_state_dict_list = [copy.deepcopy(update_model_list[i].state_dict())
                               for i in range(self.client_num)]

        init_model_dict = self.init_model.state_dict()

        base_para_matrix = torch.zeros((self.client_num,self.client_num))
        head_para_matrix = torch.zeros((self.client_num,self.client_num))
        base_para_matrix_final_last = base_para_matrix
        base_para_matrix_final = torch.zeros((self.client_num,self.client_num))

        from collections import defaultdict
        divergence_dict = dict()
        for key_name in init_model_dict.keys():
            share_para = init_model_dict[key_name]
            diver_matrix = torch.zeros((self.client_num, self.client_num))
            for model_index in range(self.client_num):
                cur_param_dict = cur_state_dict_list[model_index]
                cur_para = cur_param_dict[key_name]
                for t_model_index in range(self.client_num):
                    tmp_param_dict = cur_state_dict_list[t_model_index]
                    tmp_para = tmp_param_dict[key_name]
                    diver_matrix[model_index, t_model_index] = torch.mean(
                        (cur_para - share_para) * (tmp_para - share_para)) \
                                                               / (torch.sqrt(
                        torch.mean((cur_para - share_para) ** 2)) * torch.sqrt(
                        torch.mean((tmp_para - share_para) ** 2)))
            divergence_dict[key_name] = diver_matrix

        base_para_layer_count = 0
        head_para_layer_count = 0
        layer_weight_dict = {'conv1.weight': 0.4,
                             'conv2.weight': 0.2,
                             'fc1.weight': 0.2,
                             'fc2.weight': 0.2
                             }
        for key_name,matrix in divergence_dict.items():

            if 'bias' in key_name:
                continue
            base_para_layer_count += 1
            base_para_matrix += matrix * layer_weight_dict[key_name]

        # self.base_para_matrix = base_para_matrix*2
        # self.head_para_matrix = head_para_matrix*6

        """
        test best method for a update
        
        
        
        """
        # a = self.get_current_alpha()
        a = 0.6
        print("a:",a)
        base_para_matrix_final = a * self.base_para_matrix + (1 - a) * base_para_matrix
        head_para_matrix_final = a * self.head_para_matrix + (1 - a) * head_para_matrix


        print("self.base para matrix11")
        print(self.base_para_matrix)
        print("self.head para matrix11")
        print(self.head_para_matrix)

        # self.base_para_matrix = base_para_matrix
        # self.head_para_matrix = base_para_matrix

        self.base_para_matrix = base_para_matrix_final
        self.head_para_matrix = head_para_matrix_final

        print("update_weight_matrix_one:")
        # print(divergence_dict)
        print("base para matrix")
        print(base_para_matrix)
        print("head para matrix")
        print(head_para_matrix)
        print("self.base para matrix")
        print(self.base_para_matrix)
        print("self.head para matrix")
        print(self.head_para_matrix)

    def update_weight_matrix_v1(self, update_model_grad_list):
        """
        update weight matrix v1 in
        :param update_model_grad_list:grad update value not original model parameters
        :return:None,update client relationship weight
        """
        cur_state_dict_list = [copy.deepcopy(update_model_grad_list[i].state_dict())
                               for i in range(self.client_num)]

        init_model_dict = self.init_model.state_dict()
        base_para_matrix = torch.zeros((self.client_num, self.client_num))
        head_para_matrix = torch.zeros((self.client_num, self.client_num))

        from collections import defaultdict
        divergence_dict = dict()
        for key_name in init_model_dict.keys():
            share_para = init_model_dict[key_name]
            diver_matrix = torch.zeros((self.client_num, self.client_num))
            for model_index in range(self.client_num):
                cur_param_dict = cur_state_dict_list[model_index]
                cur_para = cur_param_dict[key_name]
                for t_model_index in range(self.client_num):
                    tmp_param_dict = cur_state_dict_list[t_model_index]
                    tmp_para = tmp_param_dict[key_name]
                    diver_matrix[model_index, t_model_index] = torch.mean(cur_para * tmp_para) \
                                                               / (torch.sqrt(torch.mean((cur_para) ** 2)) * torch.sqrt(torch.mean((tmp_para) ** 2)))
            divergence_dict[key_name] = diver_matrix

        base_para_layer_count = 0
        head_para_layer_count = 0
        layer_weight_dict = {'conv1.weight': 0.5,
                             'conv2.weight': 0.3,
                             'fc1.weight': 0.2,
                             }
        for key_name, matrix in divergence_dict.items():

            if 'bias' in key_name:
                continue
            # print(key_name)
            # print(matrix)
            if 'fc2' in key_name:
                head_para_layer_count += 1
                head_para_matrix += matrix
            else:
                base_para_layer_count += 1
                base_para_matrix += matrix * layer_weight_dict[key_name]

        # self.base_para_matrix = base_para_matrix*2
        # self.head_para_matrix = head_para_matrix*6
        # #
        self.base_para_matrix = base_para_matrix
        self.head_para_matrix = head_para_matrix
        # print(divergence_dict)
        print("base para matrix")
        print(self.base_para_matrix)
        print("head para matrix")
        print(self.head_para_matrix)

    def get_norm_weight(self,weight_matrix):
        """
        softmax weight
        :return:
        """
        norm_weight_matrix = torch.exp(weight_matrix)
        sum_norm = torch.sum(norm_weight_matrix, dim=1)
        sum_norm = sum_norm.unsqueeze(dim=1)
        norm_weight_matrix = norm_weight_matrix / sum_norm
        return norm_weight_matrix

    def aggregation_op(self, update_model_grad_list):
        """
        根据权重参数对模型进行聚合
        这里的权重相关参数访问base和head两种
        code is base on aggregation_v4 aggregation_v1 version
        freeze fc2 layer and update base layer
        difference from v4 is that this version is based on gard val
        :param update_model_grad_list:input model list
        :return:
        time:20201025
        """
        final_grad_state_dict_list = [copy.deepcopy(update_model_grad_list[i].state_dict())
                                      for i in range(self.client_num)]
        recon_state_dict_list = []
        #  reconstruct org original state dict based on
        #  self.last_state_dict_list and current gard fun
        for i in range(self.client_num):
            last_model_dict = self.last_state_dict_list[i]
            cur_model_grad_dict = final_grad_state_dict_list[i]
            recon_model_dict = copy.deepcopy(last_model_dict)
            for key_name in last_model_dict.keys():
                recon_model_dict[key_name] = last_model_dict[key_name] \
                                             + cur_model_grad_dict[key_name]
            recon_state_dict_list.append(recon_model_dict)
        # print("recon state dict result is ")
        # local_state_dict = recon_state_dict_list[0]
        # print(local_state_dict["conv1.weight"][0, 0, 0, :10])

        final_state_dict_list = [copy.deepcopy(update_model_grad_list[i].state_dict())
                                      for i in range(self.client_num)]
        #  update step
        #  update weight matrix
        param_keys_list = final_state_dict_list[0].keys()
        #  get each layer para from each model_state dict and process it
        normal_base_weight_matrix = self.get_norm_weight(self.base_para_matrix)
        # print("org base weight")
        # print(self.base_para_matrix)
        # print("normal base weight")
        # print(normal_base_weight_matrix)

        normal_head_weight_matrix = self.get_norm_weight(self.head_para_matrix)
        # print("org head weight")
        # print(self.head_para_matrix)
        # print("normal head weight")
        # print(normal_head_weight_matrix)

        for key_name in param_keys_list:

            if 'fc2' in key_name:
                normal_weight_matrix = normal_head_weight_matrix
            else:
                normal_weight_matrix = normal_base_weight_matrix

            current_layer_para_list = []
            for one_state_dict in recon_state_dict_list:
                current_layer_para = copy.deepcopy(one_state_dict[key_name])
                current_layer_para_list.append(current_layer_para)
            #   get update weight
            for model_index in range(self.client_num):
                if "fc2" in key_name:
                    cur_update_para = current_layer_para_list[model_index]
                    final_state_dict_list[model_index][key_name].copy_(cur_update_para)
                else:
                    cur_weight_list = normal_weight_matrix[model_index, :]
                    cur_update_para = sum([w * v for (w, v) in zip(cur_weight_list, current_layer_para_list)])
                    final_state_dict_list[model_index][key_name].copy_(cur_update_para)

        # print("recon state dict after weight merger result is ")
        # local_state_dict = final_state_dict_list[0]
        # print(local_state_dict["conv1.weight"][0, 0, 0, :10])

        #  calculate update grad
        send_update_grad = copy.deepcopy(final_state_dict_list)
        for key_name in param_keys_list:
            for model_index in range(self.client_num):
                send_update_grad[model_index][key_name] = final_state_dict_list[model_index][key_name] \
                                                          - recon_state_dict_list[model_index][key_name]

        # print("recon state result is ")
        # local_state_dict = recon_state_dict_list[0]
        # print(local_state_dict["conv1.weight"][0, 0, 0, :10])


        #  compress  update grad
        # for model_index in range(self.client_num):
        #     send_update_grad_dict = send_update_grad[model_index]
        #     com_send_grad_dict = self.compress_module.compress(send_update_grad_dict)
        #     recon_send_grad_dict = self.compress_module.reconstruct(com_send_grad_dict)
        #     send_update_grad[model_index] = recon_send_grad_dict

        import time
        from multiprocessing import Pool
        recon_para_call_list = []  # list that save AsyncResult object for getting real return value
        recon_para_list_r = []  # list that save real return value
        start = time.time()
        with Pool(4) as pool:
            for i in range(len(send_update_grad)):
                one_send_para_dict = send_update_grad[i]
                # convert torch data type to numpy data type
                # in order to use python multiprocess
                for k, v in one_send_para_dict.items():
                    one_send_para_dict[k] = v.cpu().numpy()
                recon_para_call_list.append(pool.apply_async(compress_fun,
                                                             args=(self.compress_module, one_send_para_dict,)))
            pool.close()
            pool.join()
            for n_index in range(len(send_update_grad)):
                recon_para_list_r.append(recon_para_call_list[n_index].get())
        #  convert numpy data type to torch data type
        #  reload it to send para and get
        for i in range(len(send_update_grad)):
            recon_grad_dict = recon_para_list_r[i]
            for k, v in recon_grad_dict.items():
                recon_grad_dict[k] = torch.tensor(v).type(torch.FloatTensor).to(device)
            send_update_grad[i] = recon_grad_dict

        #  reset last step update grad
        for key_name in param_keys_list:
            for model_index in range(self.client_num):
                self.last_state_dict_list[model_index][key_name] = recon_state_dict_list[model_index][key_name] \
                                                                   + send_update_grad[model_index][key_name]
        end = time.time()
        print("compress module use time in server is {}".format(end-start))

        # print("recon state dict after weight merger result is ")
        # local_state_dict = self.last_state_dict_list[0]
        # print(local_state_dict["conv1.weight"][0, 0, 0, :10])

        return send_update_grad

    def aggregation_op_v1(self):
        """
        根据权重参数对模型进行聚合
        这里的权重相关参数访问base和head两种
        :param update_model_list:input model list
        :return:
        time:20201014
        """
        final_state_dict_list = [copy.deepcopy(self.last_state_dict_list[i])
                                 for i in range(self.client_num)]
        #  update step
        #  update weight matrix
        param_keys_list = final_state_dict_list[0].keys()
        #  get each layer para from each model_state dict and process it
        normal_base_weight_matrix = self.get_norm_weight(self.base_para_matrix)
        print("org base weight")
        print(self.base_para_matrix)
        print("normal base weight")
        print(normal_base_weight_matrix)

        normal_head_weight_matrix = self.get_norm_weight(self.head_para_matrix)
        print("org head weight")
        print(self.head_para_matrix)
        print("normal head weight")
        print(normal_head_weight_matrix)

        for key_name in param_keys_list:

            if 'fc2' in key_name:
                normal_weight_matrix = normal_head_weight_matrix
            else:
                normal_weight_matrix = normal_base_weight_matrix

            current_layer_para_list = []
            for one_state_dict in self.last_state_dict_list:
                current_layer_para = one_state_dict[key_name]
                current_layer_para_list.append(current_layer_para)
            #   get update weight
            for model_index in range(self.client_num):
                cur_weight_list = normal_weight_matrix[model_index, :]
                cur_update_para = sum([w * v for (w, v) in zip(cur_weight_list, current_layer_para_list)])
                final_state_dict_list[model_index][key_name].copy_(cur_update_para)

        return final_state_dict_list

    def aggregation_op_v2(self,update_model_grad_list):
        """
        copy from aggregation_op debug for fc2 agg
        :param update_model_grad_list:
        :return:
        time:20201205
        """
        final_grad_state_dict_list = [copy.deepcopy(update_model_grad_list[i].state_dict())
                                      for i in range(self.client_num)]
        recon_state_dict_list = []
        #  reconstruct org original state dict based on
        #  self.last_state_dict_list and current gard fun
        for i in range(self.client_num):
            last_model_dict = self.last_state_dict_list[i]
            cur_model_grad_dict = final_grad_state_dict_list[i]
            recon_model_dict = copy.deepcopy(last_model_dict)
            for key_name in last_model_dict.keys():
                recon_model_dict[key_name] = last_model_dict[key_name] \
                                             + cur_model_grad_dict[key_name]
            recon_state_dict_list.append(recon_model_dict)
        # print("recon state dict result is ")
        # local_state_dict = recon_state_dict_list[0]
        # print(local_state_dict["conv1.weight"][0, 0, 0, :10])

        final_state_dict_list = [copy.deepcopy(update_model_grad_list[i].state_dict())
                                 for i in range(self.client_num)]
        #  update step
        #  update weight matrix
        param_keys_list = final_state_dict_list[0].keys()
        #  get each layer para from each model_state dict and process it
        normal_base_weight_matrix = self.get_norm_weight(self.base_para_matrix)
        # print("org base weight")
        # print(self.base_para_matrix)
        # print("normal base weight")
        # print(normal_base_weight_matrix)

        normal_head_weight_matrix = self.get_norm_weight(self.head_para_matrix)
        # print("org head weight")
        # print(self.head_para_matrix)
        # print("normal head weight")
        # print(normal_head_weight_matrix)

        for key_name in param_keys_list:

            if 'fc2' in key_name:
                normal_weight_matrix = normal_head_weight_matrix
            else:
                normal_weight_matrix = normal_base_weight_matrix

            current_layer_para_list = []
            for one_state_dict in recon_state_dict_list:
                current_layer_para = copy.deepcopy(one_state_dict[key_name])
                current_layer_para_list.append(current_layer_para)
            #   get update weight
            for model_index in range(self.client_num):
                # if "fc2" in key_name:
                #     cur_update_para = current_layer_para_list[model_index]
                #     final_state_dict_list[model_index][key_name].copy_(cur_update_para)
                # else:
                #     cur_weight_list = normal_weight_matrix[model_index, :]
                #     cur_update_para = sum([w * v for (w, v) in zip(cur_weight_list, current_layer_para_list)])
                #     final_state_dict_list[model_index][key_name].copy_(cur_update_para)
                cur_weight_list = normal_weight_matrix[model_index, :]
                cur_update_para = sum([w * v for (w, v) in zip(cur_weight_list, current_layer_para_list)])
                final_state_dict_list[model_index][key_name].copy_(cur_update_para)

        # print("recon state dict after weight merger result is ")
        # local_state_dict = final_state_dict_list[0]
        # print(local_state_dict["conv1.weight"][0, 0, 0, :10])

        #  calculate update grad
        send_update_grad = copy.deepcopy(final_state_dict_list)
        for key_name in param_keys_list:
            for model_index in range(self.client_num):
                send_update_grad[model_index][key_name] = final_state_dict_list[model_index][key_name] \
                                                          - recon_state_dict_list[model_index][key_name]

        # print("recon state result is ")
        # local_state_dict = recon_state_dict_list[0]
        # print(local_state_dict["conv1.weight"][0, 0, 0, :10])

        #  compress  update grad
        # for model_index in range(self.client_num):
        #     send_update_grad_dict = send_update_grad[model_index]
        #     com_send_grad_dict = self.compress_module.compress(send_update_grad_dict)
        #     recon_send_grad_dict = self.compress_module.reconstruct(com_send_grad_dict)
        #     send_update_grad[model_index] = recon_send_grad_dict

        import time
        from multiprocessing import Pool
        recon_para_call_list = []  # list that save AsyncResult object for getting real return value
        recon_para_list_r = []  # list that save real return value
        start = time.time()
        with Pool(4) as pool:
            for i in range(len(send_update_grad)):
                one_send_para_dict = send_update_grad[i]
                # convert torch data type to numpy data type
                # in order to use python multiprocess
                for k, v in one_send_para_dict.items():
                    one_send_para_dict[k] = v.cpu().numpy()
                recon_para_call_list.append(pool.apply_async(compress_fun,
                                                             args=(self.compress_module, one_send_para_dict,)))
            pool.close()
            pool.join()
            for n_index in range(len(send_update_grad)):
                recon_para_list_r.append(recon_para_call_list[n_index].get())
        #  convert numpy data type to torch data type
        #  reload it to send para and get
        for i in range(len(send_update_grad)):
            recon_grad_dict = recon_para_list_r[i]
            for k, v in recon_grad_dict.items():
                recon_grad_dict[k] = torch.tensor(v).type(torch.FloatTensor).to(device)
            send_update_grad[i] = recon_grad_dict

        #  reset last step update grad
        for key_name in param_keys_list:
            for model_index in range(self.client_num):
                self.last_state_dict_list[model_index][key_name] = recon_state_dict_list[model_index][key_name] \
                                                                   + send_update_grad[model_index][key_name]
        end = time.time()
        print("compress module use time in server is {}".format(end - start))

        # print("recon state dict after weight merger result is ")
        # local_state_dict = self.last_state_dict_list[0]
        # print(local_state_dict["conv1.weight"][0, 0, 0, :10])

        return send_update_grad

    def aggregation_op_v3(self,update_model_grad_list):
        """
        copy from aggregation_op debug for fc2 agg
        :param update_model_grad_list:
        :return:
        time:20201205
        """
        final_grad_state_dict_list = [copy.deepcopy(update_model_grad_list[i].state_dict())
                                      for i in range(self.client_num)]
        recon_state_dict_list = []
        #  reconstruct org original state dict based on
        #  self.last_state_dict_list and current gard fun
        for i in range(self.client_num):
            last_model_dict = self.last_state_dict_list[i]
            cur_model_grad_dict = final_grad_state_dict_list[i]
            recon_model_dict = copy.deepcopy(last_model_dict)
            for key_name in last_model_dict.keys():
                recon_model_dict[key_name] = last_model_dict[key_name] \
                                             + cur_model_grad_dict[key_name]
            recon_state_dict_list.append(recon_model_dict)
        # print("recon state dict result is ")
        # local_state_dict = recon_state_dict_list[0]
        # print(local_state_dict["conv1.weight"][0, 0, 0, :10])

        final_state_dict_list = [copy.deepcopy(update_model_grad_list[i].state_dict())
                                 for i in range(self.client_num)]
        #  update step
        #  update weight matrix
        param_keys_list = final_state_dict_list[0].keys()
        #  get each layer para from each model_state dict and process it

        base_para_matrix = self.base_para_matrix
        head_para_matrix = self.head_para_matrix

        import numpy as np
        cur_weight = self.cos_max_val*np.cos(2*3.14/self.local_round_num*self.local_stage_index)+self.cos_max_val
        base_para_matrix = cur_weight*base_para_matrix
        head_para_matrix = cur_weight*head_para_matrix

        normal_base_weight_matrix = self.get_norm_weight(base_para_matrix)
        # print("org base weight")
        # print(self.base_para_matrix)
        # print("normal base weight")
        # print(normal_base_weight_matrix)

        normal_head_weight_matrix = self.get_norm_weight(head_para_matrix)
        # print("org head weight")
        # print(self.head_para_matrix)
        # print("normal head weight")
        # print(normal_head_weight_matrix)

        for key_name in param_keys_list:

            if 'fc2' in key_name:
                normal_weight_matrix = normal_head_weight_matrix
            else:
                normal_weight_matrix = normal_base_weight_matrix

            current_layer_para_list = []
            for one_state_dict in recon_state_dict_list:
                current_layer_para = copy.deepcopy(one_state_dict[key_name])
                current_layer_para_list.append(current_layer_para)
            #   get update weight
            for model_index in range(self.client_num):
                # if "fc2" in key_name:
                #     cur_update_para = current_layer_para_list[model_index]
                #     final_state_dict_list[model_index][key_name].copy_(cur_update_para)
                # else:
                #     cur_weight_list = normal_weight_matrix[model_index, :]
                #     cur_update_para = sum([w * v for (w, v) in zip(cur_weight_list, current_layer_para_list)])
                #     final_state_dict_list[model_index][key_name].copy_(cur_update_para)
                cur_weight_list = normal_weight_matrix[model_index, :]
                cur_update_para = sum([w * v for (w, v) in zip(cur_weight_list, current_layer_para_list)])
                final_state_dict_list[model_index][key_name].copy_(cur_update_para)

        # print("recon state dict after weight merger result is ")
        # local_state_dict = final_state_dict_list[0]
        # print(local_state_dict["conv1.weight"][0, 0, 0, :10])

        #  calculate update grad
        send_update_grad = copy.deepcopy(final_state_dict_list)
        for key_name in param_keys_list:
            for model_index in range(self.client_num):
                send_update_grad[model_index][key_name] = final_state_dict_list[model_index][key_name] \
                                                          - recon_state_dict_list[model_index][key_name]

        # print("recon state result is ")
        # local_state_dict = recon_state_dict_list[0]
        # print(local_state_dict["conv1.weight"][0, 0, 0, :10])

        #  compress  update grad
        # for model_index in range(self.client_num):
        #     send_update_grad_dict = send_update_grad[model_index]
        #     com_send_grad_dict = self.compress_module.compress(send_update_grad_dict)
        #     recon_send_grad_dict = self.compress_module.reconstruct(com_send_grad_dict)
        #     send_update_grad[model_index] = recon_send_grad_dict

        import time
        from multiprocessing import Pool
        recon_para_call_list = []  # list that save AsyncResult object for getting real return value
        recon_para_list_r = []  # list that save real return value
        start = time.time()
        with Pool(4) as pool:
            for i in range(len(send_update_grad)):
                one_send_para_dict = send_update_grad[i]
                # convert torch data type to numpy data type
                # in order to use python multiprocess
                for k, v in one_send_para_dict.items():
                    one_send_para_dict[k] = v.cpu().numpy()
                recon_para_call_list.append(pool.apply_async(compress_fun,
                                                             args=(self.compress_module, one_send_para_dict,)))
            pool.close()
            pool.join()
            for n_index in range(len(send_update_grad)):
                recon_para_list_r.append(recon_para_call_list[n_index].get())
        #  convert numpy data type to torch data type
        #  reload it to send para and get
        for i in range(len(send_update_grad)):
            recon_grad_dict = recon_para_list_r[i]
            for k, v in recon_grad_dict.items():
                recon_grad_dict[k] = torch.tensor(v).type(torch.FloatTensor).to(device)
            send_update_grad[i] = recon_grad_dict

        #  reset last step update grad
        for key_name in param_keys_list:
            for model_index in range(self.client_num):
                self.last_state_dict_list[model_index][key_name] = recon_state_dict_list[model_index][key_name] \
                                                                   + send_update_grad[model_index][key_name]
        end = time.time()
        print("compress module use time in server is {}".format(end - start))

        # print("recon state dict after weight merger result is ")
        # local_state_dict = self.last_state_dict_list[0]
        # print(local_state_dict["conv1.weight"][0, 0, 0, :10])

        return send_update_grad

    def aggregation_op_v4(self,update_model_grad_list):
        """
        copy from aggregation_op debug for fc2 agg
        :param update_model_grad_list:
        :return:
        time:20201205
        """

        final_grad_state_dict_list = [copy.deepcopy(update_model_grad_list[i].state_dict())
                                      for i in range(self.client_num)]
        recon_state_dict_list = []
        #  reconstruct org original state dict based on
        #  self.last_state_dict_list and current gard fun
        for i in range(self.client_num):                           #服务器上进行所有client的融合
            last_model_dict = self.last_state_dict_list[i]
            cur_model_grad_dict = final_grad_state_dict_list[i]
            recon_model_dict = copy.deepcopy(last_model_dict)
            for key_name in last_model_dict.keys():
                recon_model_dict[key_name] = last_model_dict[key_name] \
                                             + cur_model_grad_dict[key_name]
            recon_state_dict_list.append(recon_model_dict)    #所有客户端更新后的梯度list

        final_state_dict_list = [copy.deepcopy(update_model_grad_list[i].state_dict())
                                 for i in range(self.client_num)]
        real_grad_state_list = [copy.deepcopy(update_model_grad_list[i].state_dict())
                                 for i in range(self.client_num)]
        #  update step
        #  update weight matrix
        param_keys_list = final_state_dict_list[0].keys()
        #  get each layer para from each model_state dict and process it
        base_para_matrix = self.base_para_matrix
        head_para_matrix = self.head_para_matrix
        normal_base_weight_matrix = self.get_norm_weight(base_para_matrix)
        normal_head_weight_matrix = self.get_norm_weight(head_para_matrix)

        for key_name in param_keys_list:
            if 'fc2' in key_name:
                normal_weight_matrix = normal_head_weight_matrix
            else:
                normal_weight_matrix = normal_base_weight_matrix

            current_layer_grad_list = []
            for one_grad_dict in final_grad_state_dict_list:      #本地更新后的client:update_model_grad_list
                current_layer_grad = copy.deepcopy(one_grad_dict[key_name])
                current_layer_grad_list.append(current_layer_grad)
            for model_index in range(self.client_num):
                cur_weight_list = normal_weight_matrix[model_index,:]   # 第i个客户端的权重矩阵
                cur_update_grad = sum([w*v for (w,v) in zip(cur_weight_list,current_layer_grad_list)])
                real_grad_state_list[model_index][key_name].copy_(cur_update_grad)

        # get final state dict
        for i in range(self.client_num):
            last_model_dict = self.last_state_dict_list[i]
            cur_model_grad_dict = real_grad_state_list[i]    #融合了相似性权重的
            recon_model_dict = final_state_dict_list[i]
            for key_name in last_model_dict.keys():
                recon_model_dict[key_name] = last_model_dict[key_name] \
                                             + cur_model_grad_dict[key_name]

        #  calculate update grad
        send_update_grad = copy.deepcopy(final_state_dict_list)
        for key_name in param_keys_list:
            for model_index in range(self.client_num):
                send_update_grad[model_index][key_name] = final_state_dict_list[model_index][key_name] \
                                                          - recon_state_dict_list[model_index][key_name]

        # print("recon state result is ")
        # local_state_dict = recon_state_dict_list[0]
        # print(local_state_dict["conv1.weight"][0, 0, 0, :10])

        #  compress  update grad
        # for model_index in range(self.client_num):
        #     send_update_grad_dict = send_update_grad[model_index]
        #     com_send_grad_dict = self.compress_module.compress(send_update_grad_dict)
        #     recon_send_grad_dict = self.compress_module.reconstruct(com_send_grad_dict)
        #     send_update_grad[model_index] = recon_send_grad_dict

        import time
        from multiprocessing import Pool
        recon_para_call_list = []  # list that save AsyncResult object for getting real return value
        recon_para_list_r = []  # list that save real return value
        start = time.time()
        with Pool(4) as pool:
            for i in range(len(send_update_grad)):
                one_send_para_dict = send_update_grad[i]
                # convert torch data type to numpy data type
                # in order to use python multiprocess
                for k, v in one_send_para_dict.items():
                    one_send_para_dict[k] = v.cpu().numpy()
                recon_para_call_list.append(pool.apply_async(compress_fun,
                                                             args=(self.compress_module, one_send_para_dict,)))
            pool.close()
            pool.join()
            for n_index in range(len(send_update_grad)):
                recon_para_list_r.append(recon_para_call_list[n_index].get())
        #  convert numpy data type to torch data type
        #  reload it to send para and get
        for i in range(len(send_update_grad)):
            recon_grad_dict = recon_para_list_r[i]
            for k, v in recon_grad_dict.items():
                recon_grad_dict[k] = torch.tensor(v).type(torch.FloatTensor).to(device)
            send_update_grad[i] = recon_grad_dict

        #  reset last step update grad
        for key_name in param_keys_list:
            for model_index in range(self.client_num):
                self.last_state_dict_list[model_index][key_name] = recon_state_dict_list[model_index][key_name] \
                                                                   + send_update_grad[model_index][key_name]
        end = time.time()
        print("compress module use time in server is {}".format(end - start))

        # print("recon state dict after weight merger result is ")
        # local_state_dict = self.last_state_dict_list[0]
        # print(local_state_dict["conv1.weight"][0, 0, 0, :10])

        return send_update_grad

    def aggregation_op_v5(self,update_model_grad_list):
        """
        copy from aggregation_op debug for fc2 agg
        freeze last layer
        :param update_model_grad_list:
        :return:
        time:20210305
        """

        final_grad_state_dict_list = [copy.deepcopy(update_model_grad_list[i].state_dict())
                                      for i in range(self.client_num)]
        recon_state_dict_list = []
        #  reconstruct org original state dict based on
        #  self.last_state_dict_list and current gard fun
        for i in range(self.client_num):
            last_model_dict = self.last_state_dict_list[i]
            cur_model_grad_dict = final_grad_state_dict_list[i]
            recon_model_dict = copy.deepcopy(last_model_dict)
            for key_name in last_model_dict.keys():
                recon_model_dict[key_name] = last_model_dict[key_name] \
                                             + cur_model_grad_dict[key_name]
            recon_state_dict_list.append(recon_model_dict)

        final_state_dict_list = [copy.deepcopy(update_model_grad_list[i].state_dict())
                                 for i in range(self.client_num)]
        real_grad_state_list = [copy.deepcopy(update_model_grad_list[i].state_dict())
                                 for i in range(self.client_num)]
        #  update step
        #  update weight matrix
        param_keys_list = final_state_dict_list[0].keys()
        #  get each layer para from each model_state dict and process it
        base_para_matrix = self.base_para_matrix
        head_para_matrix = self.head_para_matrix

        normal_base_weight_matrix = self.get_norm_weight(base_para_matrix)
        # normal_head_weight_matrix = self.get_norm_weight(head_para_matrix)
        normal_head_weight_matrix = torch.eye(head_para_matrix.shape[0])


        for key_name in param_keys_list:
            if 'fc2' in key_name:
                normal_weight_matrix = normal_head_weight_matrix
            else:
                normal_weight_matrix = normal_base_weight_matrix

            current_layer_grad_list = []
            for one_grad_dict in final_grad_state_dict_list:
                current_layer_grad = copy.deepcopy(one_grad_dict[key_name])
                current_layer_grad_list.append(current_layer_grad)
            for model_index in range(self.client_num):
                cur_weight_list = normal_weight_matrix[model_index,:]
                cur_update_grad = sum([w*v for (w,v) in zip(cur_weight_list,current_layer_grad_list)])
                real_grad_state_list[model_index][key_name].copy_(cur_update_grad)

        # get final state dict
        for i in range(self.client_num):
            last_model_dict = self.last_state_dict_list[i]
            cur_model_grad_dict = real_grad_state_list[i]
            recon_model_dict = final_state_dict_list[i]
            for key_name in last_model_dict.keys():
                recon_model_dict[key_name] = last_model_dict[key_name] \
                                             + cur_model_grad_dict[key_name]

        #  calculate update grad
        send_update_grad = copy.deepcopy(final_state_dict_list)
        for key_name in param_keys_list:
            for model_index in range(self.client_num):
                send_update_grad[model_index][key_name] = final_state_dict_list[model_index][key_name] \
                                                          - recon_state_dict_list[model_index][key_name]

        # print("recon state result is ")
        # local_state_dict = recon_state_dict_list[0]
        # print(local_state_dict["conv1.weight"][0, 0, 0, :10])

        #  compress  update grad
        # for model_index in range(self.client_num):
        #     send_update_grad_dict = send_update_grad[model_index]
        #     com_send_grad_dict = self.compress_module.compress(send_update_grad_dict)
        #     recon_send_grad_dict = self.compress_module.reconstruct(com_send_grad_dict)
        #     send_update_grad[model_index] = recon_send_grad_dict

        import time
        from multiprocessing import Pool
        recon_para_call_list = []  # list that save AsyncResult object for getting real return value
        recon_para_list_r = []  # list that save real return value
        start = time.time()
        with Pool(4) as pool:
            for i in range(len(send_update_grad)):
                one_send_para_dict = send_update_grad[i]
                # convert torch data type to numpy data type
                # in order to use python multiprocess
                for k, v in one_send_para_dict.items():
                    one_send_para_dict[k] = v.cpu().numpy()
                recon_para_call_list.append(pool.apply_async(compress_fun,
                                                             args=(self.compress_module, one_send_para_dict,)))
            pool.close()
            pool.join()
            for n_index in range(len(send_update_grad)):
                recon_para_list_r.append(recon_para_call_list[n_index].get())
        #  convert numpy data type to torch data type
        #  reload it to send para and get
        for i in range(len(send_update_grad)):
            recon_grad_dict = recon_para_list_r[i]
            for k, v in recon_grad_dict.items():
                recon_grad_dict[k] = torch.tensor(v).type(torch.FloatTensor).to(device)
            send_update_grad[i] = recon_grad_dict

        #  reset last step update grad
        for key_name in param_keys_list:
            for model_index in range(self.client_num):
                self.last_state_dict_list[model_index][key_name] = recon_state_dict_list[model_index][key_name] \
                                                                   + send_update_grad[model_index][key_name]
        end = time.time()
        print("compress module use time in server is {}".format(end - start))

        # print("recon state dict after weight merger result is ")
        # local_state_dict = self.last_state_dict_list[0]
        # print(local_state_dict["conv1.weight"][0, 0, 0, :10])

        return send_update_grad


    def get_aggregation_result(self, update_model_list, cur_loss_list=None):
        """
        对客户端的模型进行聚合
        :param update_model_list:current local update model list
        :param cur_loss_list:cur local test loss
        :return:
        """
        final_state_dict_list = self.aggregation_op(update_model_list)
        return final_state_dict_list

    def get_aggregation_result_v1(self, update_model_list, cur_loss_list=None):
        """
        对客户端的模型进行聚合
        :param update_model_list:current local update model list
        :param cur_loss_list:cur local test loss
        :return:
        """
        final_state_dict_list = self.aggregation_op_v2(update_model_list)
        return final_state_dict_list

    def get_aggregation_result_v2(self,update_model_list,cur_loss_list=None):
        """
        aggregation all client change weight dynamic
        :param update_model_list:
        :param cur_loss_list:
        :return:
        """
        final_state_dict_list = self.aggregation_op_v3(update_model_list)
        return final_state_dict_list

    def get_aggregation_result_v3(self,update_model_list,cur_loss_list=None):
        """
        aggregation all client change weight dynamic
        :param update_model_list:
        :param cur_loss_list:
        :return:
        """
        # print("update_model_list1:",update_model_list)
        final_state_dict_list = self.aggregation_op_v4(update_model_list)
        # print("final_state_dict_list:",final_state_dict_list)
        return final_state_dict_list

    def get_aggregation_result_v4(self,update_model_list,cur_loss_list=None):
        """
        aggregation all client change weight dynamic
        do not update last layer,used for ablation experiment
        :param update_model_list:
        :param cur_loss_list:
        :return:
        """
        final_state_dict_list = self.aggregation_op_v5(update_model_list)
        return final_state_dict_list

    def get_aggregation_result_mean(self):
        """
        对客户端的模型进行聚合
        :param update_model_list:current local update model list
        :param cur_loss_list:cur local test loss
        :return:
        """
        # self.update_weight_matrix(update_model_list)
        final_state_dict_list = self.aggregation_op_v1()
        return final_state_dict_list


if __name__ == "__main__":

    import sys
    import config.cifar10_config as config
    import time

    savedStdout = sys.stdout  # 保存标准输出流

    import argparse

    parse = argparse.ArgumentParser()
    parse.add_argument("--mode", type=str, default="IID")
    parse.add_argument("--client_num", type=int, default=10)
    parse.add_argument("--class_num_for_client", type=int, default=10)
    parse.add_argument("--dataset_name", type=str, default="CIFAR10")
    parse.add_argument("--model_name", type=str, default="CNN")
    parse.add_argument("--communication_rounds", type=int, default=100)
    parse.add_argument("--local_epoch", type=int, default=1)
    parse.add_argument("--batch_size", type=int, default=128)
    parse.add_argument("--use_server", type=int, default=0)
    parse.add_argument("--re_stdout", type=int, default=0)
    #  重定向输出
    args = parse.parse_args()
    if args.re_stdout == 0:
        re_stdout = False
    else:
        re_stdout = True
    cur_arguments = ExperimentPara(mode=args.mode,
                                   client_num=args.client_num,
                                   class_num_for_client=args.class_num_for_client,
                                   dataset_name=config.DATASET_NAME,
                                   model_name=args.model_name,
                                   communication_rounds=args.communication_rounds,
                                   local_epoch=args.local_epoch,
                                   batch_size=args.batch_size,
                                   use_server=(args.use_server == 1))
    current_time = time.localtime(time.time())
    time_stamp = ("2020-{}-{}-{}-{}-Release".format(current_time[1], current_time[2], current_time[3], current_time[4]))
    arguments_str = cur_arguments.get_str_rep()
    arguments_str = arguments_str + "|" + time_stamp
    print(arguments_str)

    from model.cnn_model import Net_cifar_v1 as Net

    model_list = []
    for i in range(args.client_num):
        model_list.append(Net(output_class=10))

    aggregation_module = AggregationModule(cur_arguments)

    aggregation_module.get_aggregation_result(model_list)
    aggregation_module.get_aggregation_result(model_list)
    aggregation_module.get_aggregation_result(model_list)
