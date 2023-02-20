#!/bin/bash

num_parallel=5
num_seeds=25
num_gpus=0

# flags="--silent"
flags=""

for (( i=0; i<$((num_seeds / num_parallel)); i++))
do
    for (( c=0; c<$num_parallel; c++))
    do
        if [ $num_gpus -gt 0 ]
        then 
            export CUDA_VISIBLE_DEVICES=$((c % num_gpus))
        else
            export CUDA_VISIBLE_DEVICES=''
        fi
        s=$((num_parallel*i + c))
        printf "Running seed %s\n" "${s}"
        (python main_tres_medicare.py $flags --seed=$((s + 4*num_seeds)) --tr_reg --ratio c_ratio --rdir=results --edir cutoff --eval_every 30 --n_epochs 30) &
    done 
    wait
done 
