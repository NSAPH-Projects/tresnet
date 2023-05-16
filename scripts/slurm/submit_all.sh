#!/bin/bash

scripts=("ratio_loss_monitored"
         "ratio_loss_unmonitored"
         "ratio_loss_unmonitored_lr"
         "ratio_loss_unmonitored_dropout_lr"
         "tresnet_dropout_lr"
         "tresnet_dropout"
         "tresnet_unmonitored_lr"
         "tresnet_splines"
         "tresnet"
         "tresnet_unmonitored")

# sbatch all of them
for script in "${scripts[@]}"
do
    sbatch scripts/slurm/${script}.sh
    sleep 2  # wait 2 seconds between sbatch commands
done
