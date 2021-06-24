This code includes 2 algorithms —— SPFL and PLGP, which are Synchronous Federated Learning and Asynchronous Federated Learning, we do personalization FL on both of them.

# Environment
 - python 3.7

## install packages
`requirements.txt` is basically the exact environment of mine. Let's create a conda environment and replicate the envorinment as follows.
```bash
conda create -n Fede python==3.7
conda activate Fede
pip install -r requirements.txt
```
 
# DataSet
For both of the two experiments, We use four datasets: EMNIST，MNIST，CIFAR10，CIFAR100. The split methods of datasets is in " PLGA/SPFL/dataset/ ". 

# SPFL
This is a code for Synchronous Federated Learning, and we add personalization for the FL. We named this algorithm SPFL.
## Running the SPFL experiments
```bash
sh run_all.sh
``` 
It will run all the experiments (FedAvg(Async), FedAsync, PerFedAvg, LGA, PLGA ).

# PLGP
This is a code for Asynchronous Federated Learning, and we also add personalization for the FL. We named this algorithm PLGA.
## Running the PLGA experiments
```bash
sh run.sh
``` 
It will run all the experiments (FedAvg, FedUpdate, PerFedAvg, SPFL-w(1-step), SPFL-w, SPFL(1-step), SPFL ).





