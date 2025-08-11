#!/bin/bash

for k_select in {0..2}; do
    # MLP
    #python train.py group=$1 model=mlp levels=1 kernel_size=0 k_select=$k_select
    #python train.py group=$1 model=mlp levels=0 kernel_size=0 k_select=$k_select
    # resnet
    #python train.py group=$1 model=resnet levels=1 kernel_size=0 k_select=$k_select
    #python train.py group=$1 model=resnet levels=0 kernel_size=0 k_select=$k_select
    # U-net
    python train.py group=$1 model=unet levels=1 kernel_size=0 k_select=$k_select
    python train.py group=$1 model=unet levels=0 kernel_size=0 k_select=$k_select
done
