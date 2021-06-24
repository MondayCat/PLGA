import os


DATASET_NAME = "MNIST"
DATA_ROOT = "/home/xjin/Federated_Learning/data1"
LOG_DIR = "/home/xjin/Federated_Learning/log/seed20"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
