#!/bin/bash
num_parallel=4
num_gpus=0
num_seeds=50

set=${1:-3}    

case $set in
    1)
    # Set 1. TR test
    extra_flags="--tr_reg --ratio c_ratio --rdir=results --edir cutoff --eval_every 10 --n_epochs 40 --lr 0.0001 --ls 0.1"
    # regularizations=("")
    rdir="results/"
    ;;
    2)
    # Set 2. ERM percent
    extra_flags="--tr_reg --shift_type=percent --ratio=erm --n_epochs 20 --eval_every 20 --lr 0.0001 --ls 0.1"
    # regularizations=("")
    rdir="results/"
    ;;
    3)
    # Set 3. GPS=ratio percent
    extra_flags="--tr_reg --shift_type=percent --ratio=gps_ratio --n_epochs 20 --eval_every 20 --lr 0.0001 --ls 0.1 --batch_size 4000"
    # regularizations=("")
    rdir="results/"
    ;;
    *)
    echo "Wrong input!"
    exit 2
esac


for (( i=0; i<$((num_seeds / num_parallel)); i++))
do
    for (( c=0; c<$num_parallel; c++))
    do
        s=$((num_parallel*i + c))
        # we use CUDA_VISIBLE_DEVICES in case of gpus
        if [ $num_gpus -gt 0 ]; then export CUDA_VISIBLE_DEVICES=$((c % num_gpus)); fi
        flags="${extra_flags}"
        python main_medicare.py --seed=$((s + num_seeds)) ${flags} --rdir="${rdir}" --edir="${flags}" &
    done
    wait
done

