"""
log utils
time:20200924
"""
import os
import json
from collections import defaultdict

import torch

from arguments import ExpermentPara


class Logger:
    """
    logger save model
    sub_dir tree:
        - arguements_file
        - run_log_file
        - client_log
            -client1
                - save_model
                - dataset_info
    """
    def __init__(self,log_root_dir,
                 experment_name,
                 arguements):
        self.log_root_dir = log_root_dir
        self.experment_name = experment_name
        self.arguements = arguements

        self.sub_log_dir = os.path.join(self.log_root_dir,
                                        self.experment_name)
        if not os.path.exists(self.sub_log_dir):
            os.mkdir(self.sub_log_dir)

        #  arguements file path
        self.arguements_file = os.path.join(self.sub_log_dir,"arguements.json")
        if os.path.exists(self.arguements_file):
            self.arguements = ExpermentPara()
            self.arguements.load_from_file(self.arguements_file)
        else:
            self.arguements.write_config(self.arguements_file)
        #  run log file
        self.run_log_file = os.path.join(self.sub_log_dir,"run_log")
        self.client_log_dir = os.path.join(self.sub_log_dir,"client_log")
        if not os.path.exists(self.client_log_dir):
            os.mkdir(self.client_log_dir)
        self.logname_for_client = "client_data_log.json"
        self.init_log_dir()

    def init_log_dir(self):
        """
        init log dir
        :return:
        """
        num_client = self.arguements.client_num
        client_log_dir = self.client_log_dir
        for i in range(num_client):
            one_client_dir = os.path.join(client_log_dir, "client_{}".format(i))
            if not os.path.exists(one_client_dir):
                os.mkdir(one_client_dir)
            # model_cache_dir = os.path.join(one_client_dir, "model_checkpoint")
            # if not os.path.exists(model_cache_dir):
            #     os.mkdir(model_cache_dir)

    def save_client_data_index(self, client2index_list):
        """
        save client data index
        :param client2index_list:a dict that contain data index for different client
                                 the item type is list, one list contain one class index
        :return:
        """
        cur_json_file = os.path.join(self.client_log_dir,
                                     self.logname_for_client)
        # print("save client2index to {}".format(cur_json_file))
        str_client2index_list = defaultdict(list)
        for key in client2index_list:
            for cur_list in client2index_list[key]:
                new_str_list = [str(item) for item in cur_list]
                str_client2index_list[str(key)].append(new_str_list)
        with open(cur_json_file,"w",encoding="utf-8") as fw:
            json.dump(str_client2index_list,fw,ensure_ascii=False)

    def load_client_data_index(self):
        """
        :return:
        """
        cur_json_file = os.path.join(self.client_log_dir,
                                     self.logname_for_client)
        if not os.path.exists(cur_json_file):
            raise FileNotFoundError
        with open(cur_json_file, "r", encoding="utf-8") as fr:
            json_data = json.load(fr)
        client2index_list = defaultdict(list)
        for key in json_data:
            for cur_list in json_data[key]:
                index_list = [int(item) for item in cur_list]
                client2index_list[int(key)].append(index_list)
        return client2index_list

    def save_client_model(self, model, client_index, model_name):
        """
        save client model
        :param model:
        :param client_index:
        :param model_name:
        :return:
        """
        one_client_dir = os.path.join(self.client_log_dir, "client_{}".format(client_index))
        cur_model_path = os.path.join(one_client_dir,model_name)
        torch.save(model.state_dict(),cur_model_path)

    def load_client_model(self, model, client_index, model_name):
        """
        load client model
        :param model:
        :param client_index:
        :param model_name:
        :return:
        """
        one_client_dir = os.path.join(self.client_log_dir, "client_{}".format(client_index))
        cur_model_path = os.path.join(one_client_dir, model_name)
        if not os.path.exists(cur_model_path):
            raise FileExistsError
        model.load_state_dict(torch.load(cur_model_path))
        return model
