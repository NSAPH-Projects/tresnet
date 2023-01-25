#!/bin/bash

num_seeds=50

for (( i=0; i<$(num_seeds); i++))
do
    python main_tres_medicare.py --seed=$i --tr_reg --ratio c_ratio --rdir=results/medicare --edir/cutoff
done 
