#!/bin/bash

num_seeds=50

for (( i=0; i<$((num_seeds)); i++))
do
    python main_tres_medicare.py --seed=$i --tr_reg --ratio c_ratio --rdir=results --edir cutoff --eval_every 1 --n_epochs 20
done 
