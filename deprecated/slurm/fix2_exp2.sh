#!/bin/bash
#SBATCH -t 5:00:00
#SBATCH --ntasks-per-node=48
#SBATCH --mem 96G
#SBATCH -p serial_requeue
#SBATCH -J fexp2
#SBATCH -o slurm/logs/f2exp2-%j.out
#SBATCH -e slurm/logs/f2exp2-%j.err
#SBATCH --mail-type=ALL

# rm -rf .snakemake/locks/*
snakemake  -F --rerun-incomplete --nolock --cores 48 --configfile conf/snakemake.yaml -C experiment=fix2_exp2
