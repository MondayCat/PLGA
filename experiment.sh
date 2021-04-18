#/bin/bash

gpu_check=$(nvidia-smi -q -d MEMORY|grep -E 'Free'|head -1|awk '{print $3}')
echo $gpu_check
while [ "$gpu_check" -le 9000 ]; do
    sleep 30m
done
echo 'Start FedAvg'
output = `source start.sh FedAvg`
output = `tmux kill-server`

while [ "$gpu_check" -le 9000 ]; do
    sleep 30m
done
echo 'Start ASO'
output= `source start.sh ASO`
output = `tmux kill-server`

while [ "$gpu_check" -le 9000 ]; do
    sleep 30m
done
echo 'Start FAFed'
output = `source start.sh FAFed`
output = `tmux kill-server`