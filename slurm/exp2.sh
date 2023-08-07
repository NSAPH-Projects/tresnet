#!/bin/bash
#SBATCH -t 4:00:00
#SBATCH -n 48
#SBATCH --mem 48G
#SBATCH -p serial_requeue
#SBATCH -o slurm/logs/%j.out
#SBATCH -e slurm/logs/%j.err
#SBATCH --mail-type=ALL

# rm -rf .snakemake/locks/*
snakemake --use-conda --rerun-incomplete --cores 48 exp2_all --configfile conf/snakemake.yaml -C use_srun=true
