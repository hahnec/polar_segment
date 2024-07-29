# MLP
python train.py group=$1 model=mlp lr=1e-3 levels=1 kernel_size=0
python train.py group=$1 model=mlp lr=1e-3 levels=3 kernel_size=0
python train.py group=$1 model=mlp lr=1e-3 levels=3 kernel_size=0 method=averaging
python train.py group=$1 model=mlp lr=1e-3 levels=1 kernel_size=0
python train.py group=$1 model=mlp lr=1e-3 levels=1 kernel_size=3
python train.py group=$1 model=mlp lr=1e-3 levels=3 kernel_size=3
python train.py group=$1 model=mlp lr=1e-3 levels=3 kernel_size=3 activation=leaky
python train.py group=$1 model=mlp lr=1e-3 levels=5 kernel_size=3
python train.py group=$1 model=mlp lr=1e-3 levels=5 kernel_size=0
# U-net
python train.py group=$1 model=unet levels=1 kernel_size=0
python train.py group=$1 model=unet levels=3 kernel_size=0 method=averaging
python train.py group=$1 model=unet levels=3 kernel_size=3 activation=leaky
python train.py group=$1 model=unetpp levels=1 kernel_size=0 crop=224
# polarimetry
python train.py group=$1 model=unet levels=1 kernel_size=0 data_subfolder=polarimetry
python train.py group=$1 model=unet levels=1 kernel_size=0 data_subfolder=polarimetry
python train.py group=$1 model=unet levels=1 kernel_size=0 data_subfolder=polarimetry_PDDN
python train.py group=$1 model=mlp lr=1e-3 levels=1 kernel_size=0 data_subfolder=polarimetry_PDDN
python train.py group=$1 model=mlp lr=1e-3 levels=3 kernel_size=0 data_subfolder=polarimetry_PDDN
# resnet
python train.py group=$1 model=resnet lr=1e-5 levels=1 kernel_size=0 crop=50
python train.py group=$1 model=resnet lr=1e-5 levels=1 kernel_size=0 crop=50 data_subfolder=polarimetry_PDDN