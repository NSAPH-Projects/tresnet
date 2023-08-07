#!/bin/bash
#SBATCH -t 1:00:00
#SBATCH -n 48
#SBATCH --mem 48G
#SBATCH -p shared
#SBATCH -o slurm/logs/%j.out
#SBATCH -e slurm/logs/%j.err
#SBATCH --mail-type=ALL

rm -rf .snakemake/locks/*
snakemake exp1_all --cores 32 --configfile conf/snakemake.yaml --rerun-incomplete --use-conda -C use_srun=true
