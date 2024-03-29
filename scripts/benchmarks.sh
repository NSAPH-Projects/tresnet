num_parallel=24
num_gpus=0
num_seeds=24
seed_offset=0 # $num_seeds

datasets=("ihdp-N" "news-N" "sim-N")

extra_flags="--silent --eval_every=2000"

printf "\t***** IPW + Plugin + AIPW + TRESNET + DRNET *****\n"

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
                echo "Running --dataset ${dset} main.py --seed=$s --rdir results/benchmarks --edir aipw";
                python main.py --dataset ${dset} --seed=$s --rdir "results/benchmarks" --edir "aipw" ${flags};
                echo "Running --dataset ${dset} main.py --seed=$s --rdir results/benchmarks --edir tresnet --tr_reg";
                python main.py --dataset ${dset} --seed=$s --rdir "results/benchmarks" --edir "tresnet" --tr_reg ${flags};
                echo "Running --dataset ${dset} main.py --seed=$s --rdir results/benchmarks --edir plugin";
                python main.py --dataset ${dset} --seed=$s --rdir "results/benchmarks" --edir "plugin" --outcome_only ${flags};
                echo "Running --dataset ${dset} main.py --seed=$s --rdir results/benchmarks --edir IPW";
                python main.py --dataset ${dset} --seed=$s --rdir "results/benchmarks" --edir "ipw" --ratio_only ${flags};
               echo "Running --dataset ${dset} main.py --seed=$s --rdir results/benchmarks --edir drnet";
                python main.py --dataset ${dset} --seed=$s --rdir "results/benchmarks" --edir "drnet" --outcome_only --drnet ${flags};
               echo "Running --dataset ${dset} main.py --seed=$s --rdir results/benchmarks --edir drnet --tr_reg";
                python main.py --dataset ${dset} --seed=$s --rdir "results/benchmarks" --edir "drnet_tr" --tr_reg --drnet ${flags};
            ) &
        done
        wait
    done
done


printf "\t***** Ratio *****\n"

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
                echo "Running --dataset ${dset} main.py --seed=$s --rdir results/benchmarks --edir c_ratio --tr_reg --ratio c_ratio";
                python main.py --dataset ${dset} --seed=$s --rdir "results/benchmarks" --edir "c_ratio" --ratio c_ratio --tr_reg ${flags};
                echo "Running --dataset ${dset} main.py --seed=$s --rdir results/benchmarks --edir erm --tr_reg --ratio erm";
                python main.py --dataset ${dset} --seed=$s --rdir "results/benchmarks" --edir "erm" --ratio erm --tr_reg ${flags};
            ) &
        done
        wait
    done
done


printf "\t***** Poisson *****\n"

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
                echo "Running --dataset ${dset} main.py --seed=$((s + seed_offset)) --rdir results/benchmarks --edir non_poisson";
                python main.py --dataset ${dset} --seed=$((s + seed_offset)) --rdir "results/benchmarks" --edir "non_poisson" --count --tr_reg ${flags};
                echo "Running --dataset ${dset} main.py --seed=$((s + seed_offset)) --rdir results/benchmarks --edir poisson";
                python main.py --dataset ${dset} --seed=$((s + seed_offset)) --rdir "results/benchmarks" --edir "poisson" --poisson --count --tr_reg ${flags};
            ) &
        done
        wait
    done
done