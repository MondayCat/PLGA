"""
experment args define
"""
import os
import json


class ExpermentPara:

    def __init__(self,
                 mode="IID",
                 client_num=4,
                 class_num_for_client=6,
                 dataset_name="MNIST",
                 model_name="CNN",
                 communication_rounds=50,
                 local_epoch=1,
                 batch_size=128,
                 use_server=True,
                 ):
        assert mode in ["IID", "NIID"]
        assert client_num % 2 == 0 and client_num != 0
        assert class_num_for_client % 2 == 0 and class_num_for_client > 3
        assert dataset_name in ["MNIST", "CIFAR10", "CIFAR100", "EMNIST"], "dataset name is not right"

        self.mode = mode
        self.client_num = client_num
        self.class_num_for_client = class_num_for_client
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.communication_round = communication_rounds
        self.local_epoch = local_epoch
        self.batch_size = batch_size
        self.use_server = use_server

    def get_str_rep(self):
        """
        get str represent
        :return:
        """
        para_str = "mode={}|" \
                   "client_num={}|" \
                   "class_num={}|" \
                   "dataset_name={}|" \
                   "model_name={}|" \
                   "round={}|" \
                   "local_epoch={}|" \
                   "batch_size={}|" \
                   "use_sever={}".format(self.mode,
                                         self.client_num,
                                         self.class_num_for_client,
                                         self.dataset_name,
                                         self.model_name,
                                         self.communication_round,
                                         self.local_epoch,
                                         self.batch_size,
                                         self.use_server)
        return para_str

    def write_config(self,file_path):
        """
        write config para to file
        :return:
        """
        with open(file_path,"w",encoding="utf-8") as fw:

            json_data = dict()
            json_data["mode"] = self.mode
            json_data["client_num"] = self.client_num
            json_data["class_num_for_client"] = self.class_num_for_client
            json_data["dataset_name"] = self.dataset_name
            json_data["model_name"] = self.model_name
            json_data["communication_round"] = self.communication_round
            json_data["local_epoch"] = self.local_epoch
            json_data["batch_size"] = self.batch_size
            json_data["use_server"] = self.use_server
            json.dump(json_data,fw,ensure_ascii=False)

    def load_from_file(self, file_path):
        """
        :return:
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError
        with open(file_path,"r",encoding="utf-8") as fr:
            json_data = json.load(fr)
        self.mode = json_data["mode"]
        self.client_num = json_data["client_num"]
        self.class_num_for_client = json_data["class_num_for_client"]
        self.dataset_name = json_data["dataset_name"]
        self.model_name = json_data["model_name"]
        self.communication_round = json_data["communication_round"]
        self.local_epoch = json_data["local_epoch"]
        self.batch_size = json_data["batch_size"]
        self.use_server = json_data["use_server"]


if __name__=="__main__":
    """
    test 
    """
    cur_exp_para = ExpermentPara()
    cur_exp_para.write_config(file_path="./tmp_json")
    cur_exp_para.load_from_file(file_path="./tmp_json")
    print(cur_exp_para.get_str_rep())