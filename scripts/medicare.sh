#!/bin/bash

num_parallel=4
num_seeds=8
num_gpus=0

set=${1:-1}    

case $set in
    1)
    # Set 1. TR test
    extra_flags="--tr_reg --shift_type=cutoff"
    # regularizations=("")
    rdir="results/"
    ;;
    2)
    # Set 2. ERM percent
    extra_flags="--tr_reg --shift_type=percent --ratio=erm"
    # regularizations=("")
    rdir="results/"
    ;;
    2)
    # Set 3. GPS=ratio percent
    extra_flags="--tr_reg --shift_type=percent --ratio=erm"
    # regularizations=("")
    rdir="results/"
    ;;
    *)
    echo "Wrong input!"
    exit 2
esac

# for dset in "${datasets[@]}"
# do
#     for reg in "${regularizations[@]}"
#     do
#         printf "====== Dataset: %s, Reg: %s ======" "${dset}" "${reg}"
        for (( i=0; i<$((num_seeds / num_parallel)); i++))
        do
            for (( c=0; c<$num_parallel; c++))
            do
                s=$((num_parallel*i + c))
                # we use CUDA_VISIBLE_DEVICES in case of gpus
                if [ $num_gpus -gt 0 ]; then export CUDA_VISIBLE_DEVICES=$((c % num_gpus)); fi
                flags="${extra_flags}"
                python main_tres_medicare.py --seed=$s ${flags} --rdir="${rdir}" --edir="${flags}" &
            done
            wait
        done
#     done
# done
