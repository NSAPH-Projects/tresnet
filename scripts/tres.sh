#!/bin/bash

num_parallel=20
num_seeds=20
num_gpus=0

set=${1:-1}    

case $set in
    1)
    # Set 1. TR test
    extra_flags=""
    regularizations=(""  "--tr_reg --pert simple" "--tr_reg --pert original" "--tr_reg --pert simple --detach_ratio" "--tr_reg --pert original --detach_ratio")
    rdir="results/tres_pert/"
    ;;
    2)
    # Set 2. N-grid
    extra_flags=""
    regularizations=("" "--tr_reg --n_grid 10" "--tr_reg --n_grid 25" "--tr_reg --n_grid 50")
    rdir="results/tres_ngrid/"
    ;;
    3)
    # Set 3. Reg
    extra_flags=""
    regularizations=("" "--dropout 0.0" "--tr_reg --dropout 0.0" "--tr_reg --dropout 0.05" "--tr_reg --dropout 0.2")
    rdir="results/tres_reg/"
    ;;
    4)
    # Set 4. Noise scale
    extra_flags=""
    regularizations=("--noise 0.5" "--noise 0.1" "--noise 0.05" "--noise 0.5 --tr_reg" "--noise 0.1 --tr_reg" "--noise 0.05 --tr_reg")
    rdir="results/tres_noise/"
    ;;
    5)
    # Set 5. Detach
    extra_flags=""
    regularizations=("" "--ratio erm --tr_reg" "--ratio gps_ratio --tr_reg" "--tr_reg --ratio gps_ratio --detach_ratio" "--tr_reg --ratio erm --detach_ratio")
    rdir="results/tres_detach/"
    ;;
    6)
    # Set 6. C-Ratio LS
    extra_flags=""
    regularizations=("" "--ratio c_ratio --tr_reg --ls 0.0" "--ratio c_ratio --tr_reg --ls 0.01" "--ratio c_ratio --tr_reg --ls 0.1")
    rdir="results/tres_ls/"
    ;;
    7)
    # Set 7. Detach
    extra_flags=""
    regularizations=("" "--tr_reg" "--tr_reg --ratio_norm")
    rdir="results/tres_ratio_norm/"
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
