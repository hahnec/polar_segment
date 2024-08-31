#!/bin/bash

for k_select in {1..3}; do
    # resnet
    python train.py group=$1 model=resnet lr=1e-5 levels=1 kernel_size=0 k_select=$k_select
    python train.py group=$1 model=resnet lr=1e-5 levels=0 kernel_size=0 k_select=$k_select
    # U-net
    python train.py group=$1 model=unet levels=1 kernel_size=0 k_select=$k_select
    python train.py group=$1 model=unet levels=0 kernel_size=0 k_select=$k_select
    # MLP
    python train.py group=$1 model=mlp lr=1e-3 levels=1 kernel_size=0 k_select=$k_select
    python train.py group=$1 model=mlp lr=1e-3 levels=0 kernel_size=0 k_select=$k_select
done