#!/bin/bash

num_parallel=5
num_seeds=100
num_gpus=0
num_outer_iters=$((num_seeds / num_parallel))
datasets=("ihdp-N" "sim-N")
regularizations=("" "--var_reg" "--ratio_reg" "--combo_reg")
# regularizations=("--var_reg" "--ratio_reg")
# regularizations=("" "--combo_reg")

for dset in "${datasets[@]}"
do
    for (( i=0; i<$num_outer_iters; i++))
    do
        for reg in "${regularizations[@]}"
        do
            for (( c=0; c<$num_parallel; c++))
            do
                s=$((num_parallel*i + c))
                # we use CUDA_VISIBLE_DEVICES in case of gpus
                if [ $num_gpus -gt 0 ]; then export CUDA_VISIBLE_DEVICES=$((c % num_gpus)); fi
                python main_ipw_si.py --seed=$s --dataset $dset $reg --silent &
            done
            wait
        done
    done
done
