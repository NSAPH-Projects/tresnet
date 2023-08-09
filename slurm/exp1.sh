#!/bin/bash
#SBATCH -t 6:00:00
#SBATCH --ntasks-per-node=48
#SBATCH --mem 96G
#SBATCH -p serial_requeue
#SBATCH -o slurm/logs/%j.out
#SBATCH -e slurm/logs/%j.err
#SBATCH --mail-type=ALL

source activate tresnet
snakemake -F --rerun-incomplete --nolock --cores 48 --configfile conf/snakemake.yaml -C experiment=exp1 strategy=$STRAT
