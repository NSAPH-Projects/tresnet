#!/bin/bash
#SBATCH -t 4:00:00
#SBATCH --ntasks-per-node=48
#SBATCH --mem 96G
#SBATCH -p serial_requeue
#SBATCH -J exp1
#SBATCH -o slurm/logs/exp1-%j.out
#SBATCH -e slurm/logs/exp1-%j.err
#SBATCH --mail-type=ALL

# rm -rf .snakemake/locks/*
snakemake --rerun-incomplete --cores 48 --configfile conf/snakemake.yaml -C use_srun=true experiment=exp1
