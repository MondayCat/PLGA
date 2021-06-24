#!/bin/bash
#first group experiment time:2020-5-59-22-09
#FedAvg Random seed is 80

# EMNIST IID
python experiment_fed_persional_comperssion_v9_1_1.py --mode IID --dataset_name EMNIST --client_num 10 --class_num_for_client 32 --use_server 1 --two_local_update 1 --re_stdout 1 &
python experiment_fed_persional_comperssion_v9_1_1.py --mode IID --dataset_name EMNIST --client_num 10 --class_num_for_client 32 --use_server 1 --two_local_update 0 --re_stdout 1 &
python experiment_fed_persional_comperssion_v9_3_2.py --mode IID --dataset_name EMNIST --client_num 10 --class_num_for_client 32 --use_server 1 --re_stdout 1 &
python experiment_fed_persional_comperssion_v9_3_2.py --mode IID --dataset_name EMNIST --client_num 10 --class_num_for_client 32 --use_server 1 --re_stdout 1 --two_local_update 1 &
python experiment_per_fedavg_v3.py --mode IID --dataset_name EMNIST --client_num 10 --class_num_for_client 32 --use_server 1  --re_stdout 1 &

python experiment_fedavg_v2.py --mode IID --dataset_name EMNIST --client_num 10 --class_num_for_client 32 --use_server 1  --re_stdout 1 &
python experiment_fedavg_update_v1.py --mode IID --dataset_name EMNIST --client_num 10 --class_num_for_client 32 --use_server 1  --re_stdout 1 &
wait

# EMNIST NIID
python experiment_fed_persional_comperssion_v9_1_1.py --mode NIID --dataset_name EMNIST --client_num 10 --class_num_for_client 32 --use_server 1 --two_local_update 0 --re_stdout 1 &

python experiment_fed_persional_comperssion_v9_1_1.py --mode NIID --dataset_name EMNIST --client_num 10 --class_num_for_client 32 --use_server 1 --two_local_update 1 --re_stdout 1 &
python experiment_fed_persional_comperssion_v9_3_2.py --mode NIID --dataset_name EMNIST --client_num 10 --class_num_for_client 32 --use_server 1 --re_stdout 1 &

python experiment_fed_persional_comperssion_v9_3_2.py --mode NIID --dataset_name EMNIST --client_num 10 --class_num_for_client 32 --use_server 1 --re_stdout 1 --two_local_update 1 &
python experiment_per_fedavg_v3.py --mode NIID --dataset_name EMNIST --client_num 10 --class_num_for_client 32 --use_server 1  --re_stdout 1 &

python experiment_fedavg_v2.py --mode NIID --dataset_name EMNIST --client_num 10 --class_num_for_client 32 --use_server 1  --re_stdout 1 &
python experiment_fedavg_update_v1.py --mode NIID --dataset_name EMNIST --client_num 10 --class_num_for_client 32 --use_server 1  --re_stdout 1 &
wait
# CIFAR100 IID
python experiment_fed_persional_comperssion_v9_3_2.py --mode IID --dataset_name CIFAR100 --client_num 10 --class_num_for_client 60 --use_server 1 --re_stdout 1 &


python experiment_fedavg_v2.py --mode IID --dataset_name CIFAR100 --client_num 10 --class_num_for_client 60 --use_server 1  --re_stdout 1 &

python experiment_fed_persional_comperssion_v9_1_1.py --mode IID --dataset_name CIFAR100 --client_num 10 --class_num_for_client 60 --use_server 1 --two_local_update 0 --re_stdout 1 &
python experiment_fedavg_update_v1.py --mode IID --dataset_name CIFAR100 --client_num 10 --class_num_for_client 60 --use_server 1  --re_stdout 1 &
python experiment_fed_persional_comperssion_v9_1_1.py --mode IID --dataset_name CIFAR100 --client_num 10 --class_num_for_client 60 --use_server 1 --two_local_update 1 --re_stdout 1 &
python experiment_fed_persional_comperssion_v9_3_2.py --mode IID --dataset_name CIFAR100 --client_num 10 --class_num_for_client 60 --use_server 1 --re_stdout 1 --two_local_update 1 &

python experiment_per_fedavg_v3.py --mode IID --dataset_name CIFAR100 --client_num 10 --class_num_for_client 60 --use_server 1  --re_stdout 1 &
wait
# CIFAR100 NIID
python experiment_fed_persional_comperssion_v9_3_2.py --mode NIID --dataset_name CIFAR100 --client_num 10 --class_num_for_client 60 --use_server 1 --re_stdout 1 &
python experiment_fed_persional_comperssion_v9_3_2.py --mode NIID --dataset_name CIFAR100 --client_num 10 --class_num_for_client 60 --use_server 1 --re_stdout 1 --two_local_update 1 &
python experiment_fed_persional_comperssion_v9_1_1.py --mode NIID --dataset_name CIFAR100 --client_num 10 --class_num_for_client 60 --use_server 1 --two_local_update 1 --re_stdout 1 &

python experiment_fed_persional_comperssion_v9_1_1.py --mode NIID --dataset_name CIFAR100 --client_num 10 --class_num_for_client 60 --use_server 1 --two_local_update 0 --re_stdout 1 &

python experiment_fedavg_update_v1.py --mode NIID --dataset_name CIFAR100 --client_num 10 --class_num_for_client 60 --use_server 1  --re_stdout 1 &
python experiment_fedavg_v2.py --mode NIID --dataset_name CIFAR100 --client_num 10 --class_num_for_client 60 --use_server 1  --re_stdout 1 &
python experiment_per_fedavg_v3.py --mode NIID --dataset_name CIFAR100 --client_num 10 --class_num_for_client 60 --use_server 1  --re_stdout 1 &
wait
# CIFAR10 IID
python experiment_fed_persional_comperssion_v9_1_1.py --mode IID --dataset_name CIFAR10 --client_num 10 --class_num_for_client 6 --use_server 1 --two_local_update 1 --re_stdout 1 &
python experiment_fed_persional_comperssion_v9_1_1.py --mode IID --dataset_name CIFAR10 --client_num 10 --class_num_for_client 6 --use_server 1 --two_local_update 0 --re_stdout 1 &

python experiment_fed_persional_comperssion_v9_3_2.py --mode IID --dataset_name CIFAR10 --client_num 10 --class_num_for_client 6 --use_server 1 --re_stdout 1 &
python experiment_fed_persional_comperssion_v9_3_2.py --mode IID --dataset_name CIFAR10 --client_num 10 --class_num_for_client 6 --use_server 1 --re_stdout 1 --two_local_update 1 &

python experiment_per_fedavg_v3.py --mode IID --dataset_name CIFAR10 --client_num 10 --class_num_for_client 6 --use_server 1  --re_stdout 1 &
python experiment_fedavg_v2.py --mode IID --dataset_name CIFAR10 --client_num 10 --class_num_for_client 6 --use_server 1  --re_stdout 1 &
python experiment_fedavg_update_v1.py --mode IID --dataset_name CIFAR10 --client_num 10 --class_num_for_client 6 --use_server 1  --re_stdout 1 &
wait
# CIFAR10 NIID
python experiment_fed_persional_comperssion_v9_1_1.py --mode NIID --dataset_name CIFAR10 --client_num 10 --class_num_for_client 6 --use_server 1 --two_local_update 1 --re_stdout 1 &

python experiment_fed_persional_comperssion_v9_1_1.py --mode NIID --dataset_name CIFAR10 --client_num 10 --class_num_for_client 6 --use_server 1 --two_local_update 0 --re_stdout 1 &
python experiment_fed_persional_comperssion_v9_3_2.py --mode NIID --dataset_name CIFAR10 --client_num 10 --class_num_for_client 6 --use_server 1 --re_stdout 1 &
python experiment_fed_persional_comperssion_v9_3_2.py --mode NIID --dataset_name CIFAR10 --client_num 10 --class_num_for_client 6 --use_server 1 --re_stdout 1 --two_local_update 1 &
python experiment_per_fedavg_v3.py --mode NIID --dataset_name CIFAR10 --client_num 10 --class_num_for_client 6 --use_server 1  --re_stdout 1 &

python experiment_fedavg_update_v1.py --mode NIID --dataset_name CIFAR10 --client_num 10 --class_num_for_client 6 --use_server 1  --re_stdout 1 &
python experiment_fedavg_v2.py --mode NIID --dataset_name CIFAR10 --client_num 10 --class_num_for_client 6 --use_server 1  --re_stdout 1 &
wait
# MNIST IID
python experiment_fed_persional_comperssion_v9_1_1.py --mode IID --dataset_name MNIST --client_num 10 --class_num_for_client 6 --use_server 1 --two_local_update 1  --re_stdout 1 &
python experiment_fed_persional_comperssion_v9_1_1.py --mode IID --dataset_name MNIST --client_num 10 --class_num_for_client 6 --use_server 1 --two_local_update 0  --re_stdout 1 &

python experiment_fed_persional_comperssion_v9_3_2.py --mode IID --dataset_name MNIST --client_num 10 --class_num_for_client 6 --use_server 1 --re_stdout 1 --two_local_update 1 &

python experiment_fed_persional_comperssion_v9_3_2.py --mode IID --dataset_name MNIST --client_num 10 --class_num_for_client 6 --use_server 1 --re_stdout 1 &
python experiment_per_fedavg_v3.py --mode IID --dataset_name MNIST --client_num 10 --class_num_for_client 6 --use_server 1  --re_stdout 1 &
python experiment_fedavg_update_v1.py --mode IID --dataset_name MNIST --client_num 10 --class_num_for_client 6 --use_server 1  --re_stdout 1 &

python experiment_fedavg_v2.py --mode IID --dataset_name MNIST --client_num 10 --class_num_for_client 6 --use_server 1  --re_stdout 1 &
wait
# MNIST NIID
python experiment_fed_persional_comperssion_v9_1_1.py --mode NIID --dataset_name MNIST --client_num 10 --class_num_for_client 6 --use_server 1 --two_local_update 1  --re_stdout 1 &

python experiment_fed_persional_comperssion_v9_3_2.py --mode NIID --dataset_name MNIST --client_num 10 --class_num_for_client 6 --use_server 1 --re_stdout 1 &
python experiment_per_fedavg_v3.py --mode NIID --dataset_name MNIST --client_num 10 --class_num_for_client 6 --use_server 1  --re_stdout 1 &

python experiment_fedavg_v2.py --mode NIID --dataset_name MNIST --client_num 10 --class_num_for_client 6 --use_server 1  --re_stdout 1 &
python experiment_fedavg_update_v1.py --mode NIID --dataset_name MNIST --client_num 10 --class_num_for_client 6 --use_server 1  --re_stdout 1 &
python experiment_fed_persional_comperssion_v9_1_1.py --mode NIID --dataset_name MNIST --client_num 10 --class_num_for_client 6 --use_server 1 --two_local_update 0  --re_stdout 1 &
python experiment_fed_persional_comperssion_v9_3_2.py --mode NIID --dataset_name MNIST --client_num 10 --class_num_for_client 6 --use_server 1 --re_stdout 1 --two_local_update 1 &







