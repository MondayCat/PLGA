#!/bin/sh
if [ $1 ]; then
    Algorithm=$1
else
    Algorithm='FedAvg'
fi

Dataset='MNIST'         # MNIST Cifar10 FashionMNIST
Model='cnn'             # cnn mclr
Async_process='False'    # True False
Algorithm='LGP'      # FedAvg ASO FAFed
Optimizer='SGD'         # SGD
Batch_size=64    
Lr=0.001
Lamda=0.5
Beta=0.001
Num_global_iters=800
Local_epochs=20
Num_users=10
User_labels=5
Niid='True'            # True False
Data_load='fixed'      # fixed flow
Times=1
Extra='not_drop_cross_test'

echo $Algorithm
python3 main.py --dataset=$Dataset --model=$Model --async_process=${Async_process} --batch_size=${Batch_size} --learning_rate=$Lr --lamda=$Lamda --beta=$Beta --num_global_iters=$Num_global_iters --optimizer=$Optimizer --local_epochs=$Local_epochs --algorithm=${Algorithm} --numusers=$Num_users --user_labels=$User_labels --niid=$Niid --data_load=$Data_load --times=$Times --extra=$Extra

