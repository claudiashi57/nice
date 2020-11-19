#!/usr/bin/env bash

#!/bin/bash

#IRM (the main result).
echo "hello"

HIDDEN_DIM=200
L2=0.01
LR=1e-4
STEPS=1001
ANNEAL=201
WEIGHT=100
NET=tarnet
GPU=0
ALLTRAIN=1
N=900
D=25
irm_setup=x_all
S=0


ermsetup=(
     x_and_a
     x_all
     xt
     xtat

)

echo "hi"
echo "IRM :"
  python -u main.py\
  --hidden_dim=$HIDDEN_DIM\
  --l2_regularizer_weight=$L2\
  --lr=$LR\
  --penalty_anneal_iters=$ANNEAL\
  --penalty_weight=$WEIGHT\
  --steps=$STEPS \
  --n=$N\
  --d=$D\
  --setup=$irm_setup\
  --net=$NET\
  --gpu=$GPU\
  --alltrain=$ALLTRAIN\
  --shuffle=$S
###


for setup in ${ermsetup[@]}; do
    echo "ERM:"
    python -u main.py\
      --hidden_dim=$HIDDEN_DIM\
      --l2_regularizer_weight=$L2\
      --lr=$LR\
      --penalty_anneal_iters=0\
      --penalty_weight=0.0 \
      --steps=$STEPS \
      --n=$N\
      --d=$D\
      --setup=$setup\
      --net=$NET\
      --gpu=$GPU\
      --alltrain=$ALLTRAIN\
      --shuffle=$S
done
#

#m
