#!/usr/bin/env bash

#!/bin/bash


DATA_DIR=../../dat/SpeedDatingDat/

OUTPUT_DIR=../../res/exp2/

#hyper parameters are not cross validated. they are picked as what seem reasonable.
HIDDEN_DIM=250
L2=0.0001
LR=0.001
STEPS=501
N_COL=20
NET=tarnet

mods=(
    Mod1
    Mod2
    Mod3
    Mod4
)

dims=(
    low
    med
    high
)


for i in {1..10}; do
    for mod in ${mods[@]}; do
        for dim in ${dims[@]}; do
            echo "IRM Collider:"
            python -u main.py \
              --hidden_dim=$HIDDEN_DIM \
              --l2_regularizer_weight=$L2 \
              --lr=$LR \
              --penalty_anneal_iters=201 \
              --penalty_weight=1000 \
              --steps=$STEPS \
              --mod=$mod\
              --collider=1\
              --num_col=$N_COL\
              --dimension=$dim\
              --dat=$i \
              --net=$NET\
              --data_base_dir $DATA_DIR\
              --output_base_dir $OUTPUT_DIR
##
            echo "ERM Collider:"
            python -u main.py \
              --hidden_dim=$HIDDEN_DIM  \
              --l2_regularizer_weight=$L2 \
              --lr=$LR\
              --penalty_anneal_iters=0 \
              --penalty_weight=0.0 \
              --steps=$STEPS\
              --mod=$mod\
              --collider=1\
              --num_col=$N_COL\
              --dimension=$dim\
              --dat=$i \
              --net=$NET\
              --data_base_dir $DATA_DIR\
              --output_base_dir $OUTPUT_DIR

             echo "IRM no collider:"
            python -u main.py \
              --hidden_dim=$HIDDEN_DIM \
              --l2_regularizer_weight=$L2 \
              --lr=$LR\
              --penalty_anneal_iters=401 \
              --penalty_weight=10 \
              --steps=$STEPS \
              --mod=$mod \
              --collider=0 \
              --num_col=$N_COL\
              --dimension=$dim\
              --dat=$i \
              --net=$NET\
              --data_base_dir $DATA_DIR\
              --output_base_dir $OUTPUT_DIR
##
            echo "ERM no collider:"
            python -u main.py \
              --hidden_dim=$HIDDEN_DIM  \
              --l2_regularizer_weight=$L2 \
              --lr=$LR\
              --penalty_anneal_iters=0 \
              --penalty_weight=0.0 \
              --steps=$STEPS\
              --mod=$mod\
              --collider=0\
              --num_col=$N_COL\
              --dimension=$dim\
              --dat=$i \
              --net=$NET\
              --data_base_dir $DATA_DIR\
              --output_base_dir $OUTPUT_DIR

        done
    done
done