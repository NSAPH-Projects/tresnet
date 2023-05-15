#!/bin/bash

#SBATCH --job-name=ratio_loss
#SBATCH --output=logs/slurm/ratio_loss_%A_%a.out
#SBATCH --error=logs/slurm/ratio_loss_%A_%a.err
#SBATCH --array=0-50%25
#SBATCH --time=1:00:00
#SBATCH --partition=serial_requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G

# mkdir -p logs/slurm run this command before sbatch to create the
# log directory or else sbatch will fail

flags="--silent --clean --logdir=benchmarks/ratio_loss_full --no_csv"
datasets=("ihdp" "news" "simB" "simN" "tcga-1" "tcga-2" "tcga-3")
experiments=("ipw_classifier"
             "ipw_hybrid"
             "ipw_ps"
             "ipw_multips"
             "ipw_classifier_unmonitored"
             "ipw_ps_unmonitored")
families=("gaussian" "bernoulli" "poisson")

for dset in "${datasets[@]}"
do
    for exp in "${experiments[@]}"
    do
        for fam in "${families[@]}"
        do
            echo "Running dataset: ${dset} seed: $SLURM_ARRAY_TASK_ID experiment: ${exp} family: ${fam}";
            python main.py --dataset ${dset} --seed=$SLURM_ARRAY_TASK_ID --experiment ${exp} --glm_family ${fam} ${flags};
        done
    done
done
