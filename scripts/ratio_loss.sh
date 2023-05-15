num_parallel=5
num_gpus=0
num_seeds=50

datasets=("ihdp-N" "sim-B" "news-N" "sim-N" "tcga-B")
flags="--silent --clean --logdir=benchmarks/ratio_loss"
experiments=("ipw_classifier"
             "ipw_hybrid"
             "ipw_ps"
             "ipw_multips"
             "ipw_classifier_unmonitored"
             "ipw_hybrid_unmonitored"
             "ipw_ps_unmonitored"
             "ipw_multips_unmonitored")


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
            for exp in "${experiments[@]}"
            do
                echo "Running dataset: ${dset} seed: $s experiment: ${exp}";
                # python main.py --dataset ${dset} --seed=$s --experiment ${exp} ${flags};
            done
        done
        wait
    done
done
#
