#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH --ntasks-per-node=48
#SBATCH --mem 96G
#SBATCH -p serial_requeue
#SBATCH -J exp4
#SBATCH -o slurm/logs/exp4-%j.out
#SBATCH -e slurm/logs/exp4-%j.err
#SBATCH --mail-type=ALL

# rm -rf .snakemake/locks/*
snakemake -F --rerun-incomplete --nolock --cores 48 --configfile conf/snakemake.yaml -C experiment=exp4
