#!/bin/bash
#SBATCH -t 4:00:00
#SBATCH --ntasks-per-node=48
#SBATCH --mem 96G
#SBATCH -p serial_requeue
#SBATCH -J fexp2
#SBATCH -o slurm/logs/fexp2-%j.out
#SBATCH -e slurm/logs/fexp2-%j.err
#SBATCH --mail-type=ALL

# rm -rf .snakemake/locks/*
snakemake  -F --rerun-incomplete --nolock --cores 48 --configfile conf/snakemake.yaml -C use_srun=true experiment=fix_exp2
