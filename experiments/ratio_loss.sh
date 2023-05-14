num_parallel=5
num_gpus=0
num_seeds=50
seed_offset=0

datasets=("ihdp-N" "sim-B" "news-N" "sim-N" "tcga-B")
extra_flags="--silent"
experiments=("ipw_classifier" "ipw_hybrid" "ipw")

for dset in "${datasets[@]}"
do
    printf "\t====== Dataset: %s  ======\n" "${dset}"
    for (( i=0; i<$((num_seeds / num_parallel)); i++))
    do
        for (( c=0; c<$num_parallel; c++))
        do
            s=$((num_parallel*i + c))
            if [ $num_gpus -gt 0 ]
            then 
                export CUDA_VISIBLE_DEVICES=$((c % num_gpus))
            else
                export CUDA_VISIBLE_DEVICES=''
            fi
            flags="${extra_flags}"
            (
                for exp in "${experiments[@]}"
                do
                    echo "Running --dataset ${dset} main.py --seed=$s --experiment ${exp}";
                    python main.py --dataset ${dset} --seed=$s --experiment ${exp} ${flags};
                done
            ) &
        done
        wait
    done
done
# 
