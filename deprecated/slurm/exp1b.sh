#!/bin/bash
#SBATCH -t 6:00:00
#SBATCH --ntasks-per-node=48
#SBATCH --mem 96G
#SBATCH -p serial_requeue
#SBATCH -J exp1b
#SBATCH -o slurm/logs/exp1b-%j.out
#SBATCH -e slurm/logs/exp1b-%j.err
#SBATCH --mail-type=ALL

# rm -rf .snakemake/locks/*
snakemake --rerun-incomplete --nolock --cores 48 --configfile conf/snakemake.yaml -C use_srun=true experiment=exp1b
