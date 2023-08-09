#! /bin/bash

strategies=("tresnet-clever"
            "vcnet"
            "aipw"
            "outcome")

for strat in "${strategies[@]}"; do
    o_=slurm/logs/exp2-${strat}-%j.out
    e_=slurm/logs/exp2-${strat}-%j.err
    J_=exp2-$strat
    STRAT=$strat sbatch -J ${J_} -o ${o_} -e ${e_} slurm/exp2.sh
    sleep 2  # wait 2 seconds between sbatch commands
done
