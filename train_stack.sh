# MLP
python train.py model=mlp levels=1 kernel_size=0 wlens='[550]'
python train.py model=mlp levels=3 kernel_size=0 wlens='[550]'
python train.py model=mlp levels=3 kernel_size=0 method=averaging wlens='[550]'
python train.py model=mlp levels=1 kernel_size=0 wlens='[550]'
python train.py model=mlp levels=1 kernel_size=3 wlens='[550]'
python train.py model=mlp levels=3 kernel_size=3 wlens='[550]'
python train.py model=mlp levels=3 kernel_size=3 activation=leaky wlens='[550]'
python train.py model=mlp levels=5 kernel_size=3 wlens='[550]'
python train.py model=mlp levels=5 kernel_size=0
python train.py model=mlp levels=5 kernel_size=3
# U-net
python train.py model=unet levels=1 kernel_size=0
python train.py model=unet levels=3 kernel_size=0 method=averaging
python train.py model=unet levels=3 kernel_size=3 activation=leaky
python train.py model=unetpp levels=1 kernel_size=0 crop=224
# polarimetry data
python train.py model=unet levels=1 kernel_size=0 data_subfolder=polarimetry wlens='[550]'
python train.py model=unet levels=1 kernel_size=0 data_subfolder=polarimetry wlens='[550]'
python train.py model=unet levels=1 kernel_size=0 data_subfolder=polarimetry_PDDN wlens='[550]'
python train.py model=mlp levels=1 kernel_size=0 data_subfolder=polarimetry_PDDN wlens='[550]'
python train.py model=mlp levels=3 kernel_size=0 data_subfolder=polarimetry_PDDN wlens='[550]'
