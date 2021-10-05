"""
rewrite base class
time:
"""

import os
import json


class ExperimentPara:

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

    def save_json_file(self,json_data,file_path):
        """
        save json data to dst file path
        """
        with open(file_path,"w",encoding="utf-8") as fw:
            json.dump(json_data,fw,ensure_ascii=False)

    def load_json_file(self,file_path):
        """
        load json data from dst file
        :param file_path:
        :return:
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError
        with open(file_path, "r", encoding="utf-8") as fr:
            json_data = json.load(fr)
        return json_data

    def get_json_data_from_para(self):
        """
        :return:
        """
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
        return json_data

    def set_para_from_json_data(self,json_data):
        """
        assign self
        :return:
        """
        try:
            self.mode = json_data["mode"]
            self.client_num = json_data["client_num"]
            self.class_num_for_client = json_data["class_num_for_client"]
            self.dataset_name = json_data["dataset_name"]
            self.model_name = json_data["model_name"]
            self.communication_round = json_data["communication_round"]
            self.local_epoch = json_data["local_epoch"]
            self.batch_size = json_data["batch_size"]
            self.use_server = json_data["use_server"]
        except KeyError as e:
            print("some error happen {}".format(e))
            raise KeyError

    def write_config(self,file_path):
        json_data = self.get_json_data_from_para()
        self.save_json_file(json_data,file_path)

    def load_from_file(self,file_path):
        json_data = self.load_json_file(file_path)
        self.set_para_from_json_data(json_data)

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
                   "use_sever={}|".format(self.mode,
                                          self.client_num,
                                          self.class_num_for_client,
                                          self.dataset_name,
                                          self.model_name,
                                          self.communication_round,
                                          self.local_epoch,
                                          self.batch_size,
                                          self.use_server)
        return para_str


class ExperimentParaUpdate(ExperimentPara):
    """
    
    """
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
                 local_fine_step=1
                 ):
        super(ExperimentParaUpdate, self).__init__(mode=mode,
                                                   client_num=client_num,
                                                   class_num_for_client=class_num_for_client,
                                                   dataset_name=dataset_name,
                                                   model_name=model_name,
                                                   communication_rounds=communication_rounds,
                                                   local_epoch=local_epoch,
                                                   batch_size=batch_size,
                                                   use_server=use_server)
        self.local_fine_step = local_fine_step

    def get_str_rep(self):
        """

        :return:
        """
        base_para_str = super(ExperimentParaUpdate,self).get_str_rep()
        cur_para_str = "local_fine_step={}|".format(self.local_fine_step)
        return base_para_str+cur_para_str

    def get_json_data_from_para(self):

        base_json_data = super(ExperimentParaUpdate,self).get_json_data_from_para()
        cur_json_data = dict()
        cur_json_data["local_fine_step"] = self.local_fine_step
        base_json_data.update(cur_json_data)
        return base_json_data

    def set_para_from_json_data(self,json_data):
        super(ExperimentParaUpdate, self).set_para_from_json_data(json_data)
        self.local_fine_step = json_data["local_fine_step"]

    def load_from_file(self, file_path):
        json_data = self.load_json_file(file_path)
        self.set_para_from_json_data(json_data)


class ExperimentParaCompressWithPersonal(ExperimentPara):
    """
   
    """
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
                 local_two_update = False,
                 compress_ratio = 4,
                 use_quan = False,
                 use_para_decom =False
                 ):
        super(ExperimentParaCompressWithPersonal, self).__init__(mode=mode,
                                                   client_num=client_num,
                                                   class_num_for_client=class_num_for_client,
                                                   dataset_name=dataset_name,
                                                   model_name=model_name,
                                                   communication_rounds=communication_rounds,
                                                   local_epoch=local_epoch,
                                                   batch_size=batch_size,
                                                   use_server=use_server)
        self.local_two_update = local_two_update
        self.compress_ratio = compress_ratio
        self.use_quan = use_quan
        self.use_para_decom = use_para_decom

    def get_str_rep(self):
        """

        :return:
        """
        base_para_str = super(ExperimentParaCompressWithPersonal,self).get_str_rep()
        cur_para_str = "two_update_step={}|use_quan={}|use_para_decom={}|compress_ratio={}|".format(
            self.local_two_update, self.use_quan,
            self.use_para_decom,
            self.compress_ratio)
        return base_para_str+cur_para_str

    def get_json_data_from_para(self):

        base_json_data = super(ExperimentParaCompressWithPersonal,self).get_json_data_from_para()
        cur_json_data = dict()
        cur_json_data["use_quan"] = self.use_quan
        cur_json_data["use_para_decom"] = self.use_para_decom
        cur_json_data["compress_ratio"] = self.compress_ratio
        cur_json_data["local_two_update"] = self.local_two_update
        base_json_data.update(cur_json_data)
        return base_json_data

    def set_para_from_json_data(self,json_data):

        super(ExperimentParaCompressWithPersonal, self).set_para_from_json_data(json_data)
        self.use_quan = json_data["use_quan"]
        self.use_para_decom = json_data["use_para_decom"]
        self.compress_ratio = json_data["compress_ratio"]
        self.local_two_update = json_data["local_two_update"]

    def load_from_file(self, file_path):
        json_data = self.load_json_file(file_path)
        self.set_para_from_json_data(json_data)


class ExperimentParaCompressWithPersonalAblation(ExperimentParaCompressWithPersonal):
    """
    
    """
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
                 local_two_update = False,
                 compress_ratio = 4,
                 use_quan = False,
                 use_para_decom =False,
                 rclient_num =  10,
                 update_round = 10,
                 ):
        super(ExperimentParaCompressWithPersonalAblation, self).__init__(mode=mode,
                                                   client_num=client_num,
                                                   class_num_for_client=class_num_for_client,
                                                   dataset_name=dataset_name,
                                                   model_name=model_name,
                                                   communication_rounds=communication_rounds,
                                                   local_epoch=local_epoch,
                                                   batch_size=batch_size,
                                                   use_server=use_server,
                                                   local_two_update=local_two_update,
                                                   compress_ratio=compress_ratio,
                                                   use_quan=use_quan,
                                                   use_para_decom=use_para_decom)

        self.rclient_num = rclient_num
        self.update_round = update_round

    def get_str_rep(self):
        """

        :return:
        """
        base_para_str = super(ExperimentParaCompressWithPersonalAblation,self).get_str_rep()
        cur_para_str = "rclient={}|update_round={}".format(
            self.rclient_num, self.update_round)
        return base_para_str+cur_para_str

    def get_json_data_from_para(self):

        base_json_data = super(ExperimentParaCompressWithPersonalAblation,self).get_json_data_from_para()
        cur_json_data = dict()
        cur_json_data["rclient_num"] = self.rclient_num
        cur_json_data["update_round"] = self.update_round
        base_json_data.update(cur_json_data)
        return base_json_data

    def set_para_from_json_data(self,json_data):

        super(ExperimentParaCompressWithPersonalAblation, self).set_para_from_json_data(json_data)
        self.rclient_num = json_data["rclient_num"]
        self.update_round = json_data["update_round"]

    def load_from_file(self, file_path):
        json_data = self.load_json_file(file_path)
        self.set_para_from_json_data(json_data)



class ExperimentParaCompressWithPersonalAddQuan(ExperimentParaCompressWithPersonal):
    """
    """

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
                 local_two_update=False,
                 compress_ratio=4,
                 use_quan=False,
                 use_para_decom=False,
                 quan_method="natural",
                 shift_bit=2,
                 ):
        super(ExperimentParaCompressWithPersonalAddQuan, self).__init__(mode=mode,
                                                                 client_num=client_num,
                                                                 class_num_for_client=class_num_for_client,
                                                                 dataset_name=dataset_name,
                                                                 model_name=model_name,
                                                                 communication_rounds=communication_rounds,
                                                                 local_epoch=local_epoch,
                                                                 batch_size=batch_size,
                                                                 use_server=use_server,
                                                                 local_two_update=local_two_update,
                                                                 compress_ratio=compress_ratio,
                                                                 use_quan=use_quan,
                                                                 use_para_decom=use_para_decom)
        self.quan_method = quan_method
        self.shift_bit = shift_bit

    def get_str_rep(self):
        """

        :return:
        """
        base_para_str = super(ExperimentParaCompressWithPersonalAddQuan, self).get_str_rep()
        cur_para_str = "quan_method={}|shift_bit={}|".format(
            self.quan_method, self.shift_bit)
        return base_para_str + cur_para_str

    def get_json_data_from_para(self):
        base_json_data = super(ExperimentParaCompressWithPersonalAddQuan, self).get_json_data_from_para()
        cur_json_data = dict()
        cur_json_data["quan_method"] = self.quan_method
        cur_json_data["shift_bit"] = self.shift_bit
        base_json_data.update(cur_json_data)
        return base_json_data

    def set_para_from_json_data(self, json_data):
        super(ExperimentParaCompressWithPersonalAddQuan, self).set_para_from_json_data(json_data)
        self.quan_method = json_data["quan_method"]
        self.shift_bit = json_data["shift_bit"]

    def load_from_file(self, file_path):
        json_data = self.load_json_file(file_path)
        self.set_para_from_json_data(json_data)


if __name__=="__main__":
    """
    test 
    """
    cur_exp_para = ExperimentParaCompressWithPersonalAblation()
    cur_exp_para.write_config(file_path="./tmp_json")
    cur_exp_para.load_from_file(file_path="./tmp_json")
    print(cur_exp_para.get_str_rep())
