#! /bin/bash

strategies=("tresnet-clever"
            "tresnet-clever-splines"
            "tresnet-clever-ps"
            "tresnet-clever-telescope"
            "tresnet-clever-1"
            "vcnet"
            "aipw"
            "aipw-ps"
            "tarnet"
            "tarnet-ps"
            "outcome"
            "ipw-telescope"
            "ipw-classifier"
            "ipw-ps"
            "ipw-hybrid"
            "ipw-multips"
            "tmle-clever")

for strat in "${strategies[@]}"; do
    o_=slurm/logs/exp1-${strat}-%j.out
    e_=slurm/logs/exp1-${strat}-%j.err
    J_=exp1-$strat
    STRAT=$strat sbatch -J ${J_} -o ${o_} -e ${e_} slurm/exp1.sh
    sleep 2  # wait 2 seconds between sbatch commands
done
