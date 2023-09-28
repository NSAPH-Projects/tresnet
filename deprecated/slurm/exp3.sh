#!/bin/bash
#SBATCH -t 4:00:00
#SBATCH --ntasks-per-node=48
#SBATCH --mem 96G
#SBATCH -p serial_requeue
#SBATCH -J exp3
#SBATCH -o slurm/logs/exp3-%j.out
#SBATCH -e slurm/logs/exp3-%j.err
#SBATCH --mail-type=ALL

# rm -rf .snakemake/locks/*
snakemake --rerun-incomplete --nolock --cores 48 --configfile conf/snakemake.yaml -C use_srun=false experiment=exp3
