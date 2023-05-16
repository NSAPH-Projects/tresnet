#!/bin/bash

#SBATCH --job-name=tresdoutlr
#SBATCH --output=logs/slurm/tresdoutlr_%A_%a.out
#SBATCH --error=logs/slurm/tresdoutlrp_%A_%a.err
#SBATCH --array=0-30%15
#SBATCH --time=8:00:00
#SBATCH --partition=serial_requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G


soff=0  # seed offset for additional experiments
flags="--silent --clean --logdir=benchmarks/tresnet_dropout_lr --no_csv --dropout 0.2 --lr 0.001"
datasets=("ihdp" "news" "simB" "simN" "tcga-1" "tcga-2" "tcga-3")
experiments=("aipw" "tresnet" "outcome" "ipw_classifier" "tresnet_clever")
families=("gaussian" "bernoulli" "poisson")

for dset in "${datasets[@]}"
do
    for exp in "${experiments[@]}"
    do
        for fam in "${families[@]}"
        do
            seed=$((SLURM_ARRAY_TASK_ID + soff))
            echo "Running dataset: ${dset} seed: ${seed} experiment: ${exp} family: ${fam}";
            python main.py --dataset ${dset} --seed=${seed} --experiment ${exp} --glm_family ${fam} ${flags};
        done
    done
done
