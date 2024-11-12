#!/bin/bash

for k_select in {0..3}; do
    # U-net
#    python train.py group=$1_absence model=unet levels=1 kernel_size=0 k_select=$k_select
    python train.py group=$1_absence model=unet levels=0 kernel_size=0 k_select=$k_select
#    python train.py group=$1_rota model=unet levels=1 kernel_size=0 k_select=$k_select rotation=180
    python train.py group=$1_rota model=unet levels=0 kernel_size=0 k_select=$k_select rotation=180
#    python train.py group=$1_flip model=unet levels=1 kernel_size=0 k_select=$k_select flips=1
    python train.py group=$1_flip model=unet levels=0 kernel_size=0 k_select=$k_select flips=1
#    python train.py group=$1_rotaflip model=unet levels=1 kernel_size=0 k_select=$k_select rotation=180 flips=1
    python train.py group=$1_rotaflip model=unet levels=0 kernel_size=0 k_select=$k_select rotation=180 flips=1
#    python train.py group=$1_crop model=unet levels=0 kernel_size=0 k_select=$k_select crop=1
    python train.py group=$1_noise model=unet levels=0 kernel_size=0 k_select=$k_select noise=0.1
    python train.py group=$1_gamma model=unet levels=0 kernel_size=0 k_select=$k_select gamma=1
done
