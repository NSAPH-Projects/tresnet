#!/bin/bash

#SBATCH --job-name=rlunlrd
#SBATCH --output=logs/slurm/rlunlrd_%A_%a.out
#SBATCH --error=logs/slurm/rlunlrd_%A_%a.err
#SBATCH --array=0-30%15
#SBATCH --time=8:00:00
#SBATCH --partition=serial_requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G

# mkdir -p logs/slurm run this command before sbatch to create the
# log directory or else sbatch will fail

soff=0  # seed offset for additional experiments
flags="--silent --clean --logdir=benchmarks/ratio_loss_unmonitored_dropout_lr --unmonitor --no_csv --dropout 0.2 --lr 0.001"
datasets=("ihdp" "news" "simB" "simN" "tcga-1" "tcga-2" "tcga-3")
experiments=("ipw_classifier"
             "ipw_hybrid"
             "ipw_ps"
             "ipw_multips")
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
