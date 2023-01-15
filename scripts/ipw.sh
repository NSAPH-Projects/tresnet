#!/bin/bash

num_parallel=4
num_seeds=100
num_outer_iters=$((num_seeds / num_parallel))
prog="python main_ipw_si.py"

for dset in sim-N ihdp-N
do
    for (( i=0; i<$num_outer_iters; i++))
    do
        for (( c=0; c<$num_parallel; c++))
        do
            s=$((num_parallel*i + c))
            CUDA_VISIBLE_DEVICES=$c $prog --seed=$s --dataset=$dset --silent &
        done
        wait
    done
done
