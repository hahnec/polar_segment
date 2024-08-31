# resnet
python train.py group=$1 model=resnet lr=1e-5 levels=1 kernel_size=0
python train.py group=$1 model=resnet lr=1e-5 levels=0 kernel_size=0
# U-net
python train.py group=$1 model=unet levels=1 kernel_size=0
python train.py group=$1 model=unet levels=0 kernel_size=0
# MLP
python train.py group=$1 model=mlp lr=1e-3 levels=1 kernel_size=0
python train.py group=$1 model=mlp lr=1e-3 levels=0 kernel_size=0
