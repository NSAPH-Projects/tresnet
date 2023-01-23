#!/bin/bash

num_parallel=10
num_seeds=30
num_gpus=0

set=${1:-1}    

case $set in
    1)
    # Set 1. TR test
    extra_flags="--dropout 0.05"
    regularizations=("--tr_reg --pert original --tr discrete --no_erm"  "--tr_reg --pert simple --tr discrete" "--tr_reg --pert original --tr discrete" "--tr_reg --pert original --tr discrete --ratio_normalize"   "--tr_reg --pert original --tr vc" "")
    rdir="results/tres_pert/"
    ;;
    2)
    # Set 2. N-grid
    extra_flags="--dropout 0.05"
    regularizations=("--tr_reg --n_grid 10" "--tr_reg --n_grid 20" "--tr_reg --n_grid 30")
    rdir="results/tres_ngrid/"
    ;;
    3)
    # Set 3. Ratio reg
    extra_flags=""
    regularizations=("" "--dropout 0.05" "--tr_reg" "--tr_reg --dropout 0.05" "--tr_reg --dropout 0.05  --ratio_normalize"  "--tr_reg --dropout 0.05 --pos_reg --reg_multiscale" "--dropout 0.05")
    rdir="results/tres_reg/"
    ;;
    4)
    # Set 4. Noise scale
    extra_flags="--dropout 0.05"
    regularizations=("--noise 0.5" "--noise 0.1" "--noise 0.5 --tr_reg" "--noise 0.1 --tr_reg")
    rdir="results/tres_noise/"
    ;;
    5)
    # Set 5. Other regs
    extra_flags="--reg_multiscale"
    regularizations=("" "--var_reg --tr_reg" "--tr_reg --detach_ratio" "--tr_reg" "--tr_reg --dropout 0.05" "--tr_reg --dropout 0.05 --detach_ratio")
    rdir="results/tres_other/"
    ;;
    *)
    echo "Wrong input!"
    exit 2
esac

# datasets=("news-N")
datasets=("ihdp-N" "news-N" "sim-N")


for dset in "${datasets[@]}"
do
    for reg in "${regularizations[@]}"
    do
        printf "====== Dataset: %s, Reg: %s ======" "${dset}" "${reg}"
        for (( i=0; i<$((num_seeds / num_parallel)); i++))
        do
            for (( c=0; c<$num_parallel; c++))
            do
                s=$((num_parallel*i + c))
                # we use CUDA_VISIBLE_DEVICES in case of gpus
                if [ $num_gpus -gt 0 ]; then export CUDA_VISIBLE_DEVICES=$((c % num_gpus)); fi
                flags="--dataset ${dset} ${reg} ${extra_flags}"
                python main_tres.py --seed=$s --silent --rdir ${rdir} --edir "${flags}" ${flags} &
            done
            wait
        done
    done
done
