#!/bin/bash
#SBATCH -t 5:00:00
#SBATCH --ntasks-per-node=48
#SBATCH --mem 96G
#SBATCH -p serial_requeue
#SBATCH -J fexp3
#SBATCH -o slurm/logs/f3exp2-%j.out
#SBATCH -e slurm/logs/f3exp2-%j.err
#SBATCH --mail-type=ALL

# rm -rf .snakemake/locks/*
snakemake  -F --rerun-incomplete --nolock --cores 48 --configfile conf/snakemake.yaml -C experiment=fix3_exp2
