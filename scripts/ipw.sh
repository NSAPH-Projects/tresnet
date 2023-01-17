#!/bin/bash

num_parallel=25
num_seeds=25
num_gpus=0

# rdir=results
flags="--wd 5e-3 --n_epoch 800"

datasets=("ihdp-N sim-N")
regularizations=("" "--combo_reg" "--var_reg" "--ratio_reg")

# regularizations=("--var_reg" "--ratio_reg")
# regularizations=("" "--combo_reg")

for dset in "${datasets[@]}"
do
    for reg in "${regularizations[@]}"
    do
        printf "====== Dataset: %s, Reg: %s ======" $dset $reg
        for (( i=0; i<$((num_seeds / num_parallel)); i++))
        do
            for (( c=0; c<$num_parallel; c++))
            do
                s=$((num_parallel*i + c))
                # we use CUDA_VISIBLE_DEVICES in case of gpus
                if [ $num_gpus -gt 0 ]; then export CUDA_VISIBLE_DEVICES=$((c % num_gpus)); fi
                python main_ipw_si.py --seed=$s --dataset $dset ${reg} --rdir "\"results ${flags}\"" ${flags} --silent &
            done
            wait
        done
    done
done
