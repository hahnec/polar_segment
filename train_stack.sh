# resnet
python train.py group=$1 model=resnet lr=1e-5 levels=1 kernel_size=0 epochs=200
python train.py group=$1 model=resnet lr=1e-5 levels=1 kernel_size=0 data_subfolder=polarimetry_PDDN epochs=200
# U-net
python train.py group=$1 model=unet levels=1 kernel_size=0
python train.py group=$1 model=unet levels=0 kernel_size=0  # select 10 relevant Mueller matrix channels
python train.py group=$1 model=unet levels=3 kernel_size=0 method=averaging
python train.py group=$1 model=unet levels=3 kernel_size=3 activation=leaky
python train.py group=$1 model=unetpp levels=1 kernel_size=0
python train.py group=$1 model=unetpp levels=0 kernel_size=0 # select 10 relevant Mueller matrix channels
python train.py group=$1 model=uctransnet levels=1 kernel_size=0 crop=224
python train.py group=$1 model=unet levels=1 kernel_size=0 data_subfolder=polarimetry
python train.py group=$1 model=unet levels=1 kernel_size=0 data_subfolder=polarimetry_PDDN
# MLP
python train.py group=$1 model=mlp lr=1e-3 levels=1 kernel_size=0
#python train.py group=$1 model=mlp lr=1e-3 levels=3 kernel_size=0
#python train.py group=$1 model=mlp lr=1e-3 levels=3 kernel_size=0 method=averaging
#python train.py group=$1 model=mlp lr=1e-3 levels=1 kernel_size=3
#python train.py group=$1 model=mlp lr=1e-3 levels=3 kernel_size=3
python train.py group=$1 model=mlp lr=1e-3 levels=3 kernel_size=3 activation=leaky
#python train.py group=$1 model=mlp lr=1e-3 levels=1 kernel_size=0 data_subfolder=polarimetry_PDDN
#python train.py group=$1 model=mlp lr=1e-3 levels=3 kernel_size=0 data_subfolder=polarimetry_PDDN

