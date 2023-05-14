#!/bin/bash

#SBATCH --job-name=ratio_loss
#SBATCH --output=logs/slurm/ratio_loss_%A_%a.out
#SBATCH --error=logs/slurm/ratio_loss_%A_%a.err
#SBATCH --array=0-100%25
#SBATCH --time=4:00:00
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G


datasets=("ihdp-N" "sim-B" "news-N" "sim-N" "tcga-B")
flags="--silent --clean --logdir=benchmarks/ratio_loss"
experiments=("ipw_classifier" "ipw_hybrid" "ipw_ps")

for dset in "${datasets[@]}"
do
    for exp in "${experiments[@]}"
    do
        echo "Running dataset: ${dset} seed: $SLURM_ARRAY_TASK_ID experiment: ${exp}";
        python main.py --dataset ${dset} --seed=$SLURM_ARRAY_TASK_ID --experiment ${exp} ${flags};
    done
done
