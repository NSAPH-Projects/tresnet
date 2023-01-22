#!/bin/bash

num_parallel=35
num_seeds=100
num_gpus=0

# rdir=results
flags="--n_grid 10 --reg_multiscale"
# flags="--batch_size 32"


# datasets=("news-N")
datasets=("news-N" "sim-N" "ihdp-N")
regularizations=("--tr_reg --pert simple --tr discrete" "--tr_reg --pert original --tr discrete" "--tr_reg --pert simple --tr vc" "--tr_reg --pert original --tr vc" "")

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
                python main_ipw_si.py --seed=$s --dataset ${dset} ${reg} --rdir "results/tres ${flags}" ${flags} --silent &
            done
            wait
        done
    done
done
