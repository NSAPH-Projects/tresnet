#! /bin/bash

strategies=("tresnet-clever"
            "tresnet-clever-forcemse"
            "tresnet-clever-telescope"
            "outcome"
            "outcome-forcemse")

for strat in "${strategies[@]}"; do
    o_=slurm/logs/exp3-${strat}-%j.out
    e_=slurm/logs/exp3-${strat}-%j.err
    J_=exp3-$strat
    STRAT=$strat sbatch -J ${J_} -o ${o_} -e ${e_} slurm/exp3.sh
    sleep 2  # wait 2 seconds between sbatch commands
done
