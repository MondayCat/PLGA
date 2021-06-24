"""
多个模型权重融合的模块，
第一个版本实现联邦平均算法
time:20200927
"""
import copy

import torch


class AggregationModule:
    """
    多个客户端权重融合的算法
    """
    def __init__(self,cur_arguments):

        self.cur_arguments = cur_arguments
        self.cur_model_list = None
        self.last_step_model_list = None
    #
    # def set_cur_client_models(self,model_list):
    #     self.cur_model_list = model_list

    def get_aggregation_result(self,update_model_list):
        """
        对客户端的模型进行聚合
        :param update_model_list:
        :return:
        """
        # update step
        final_state_dict = copy.deepcopy(update_model_list[0].state_dict())
        param_keys_list = final_state_dict.keys()
        # get each para  from each model_state dict and process on it
        for key_name in param_keys_list:
            current_layer_para_list = []
            for currrent_model in update_model_list:
                one_state_dict = currrent_model.state_dict()
                current_layer_para = one_state_dict[key_name]
                current_layer_para_list.append(current_layer_para)
            # this federate avg algorithm
            process_result = torch.mean(torch.stack(current_layer_para_list), 0)
            final_state_dict[key_name].copy_(process_result)
        final_state_dict_list = [final_state_dict.copy()
                                 for i in range(self.cur_arguments.client_num)]
        return final_state_dict_list


