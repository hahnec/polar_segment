from pathlib import Path
import torch
from mm.models import MuellerMatrixModel
from mm.utils.cod import read_cod_data_X3D
from utils.mm_rotation import RawRandomMuellerRotation

base_dir = Path('/media/chris/EB62-383C/CC_Rotation/')
calib_path = base_dir / '2021-10-21_C_2' / '550nm'
with open(base_dir / 'angles_and_center_points.txt', 'r') as f:
    lines = f.readlines()
transforms = torch.tensor([[float(el) for el in line.strip('\n').split(' ')] for line in lines])
A = read_cod_data_X3D(str(calib_path / '550_A.cod'))
W = read_cod_data_X3D(str(calib_path / '550_W.cod'))
feat_keys = ['azimuth']
mm_model = MuellerMatrixModel(bA=A[None], bW=W[None], feature_keys=feat_keys, wnum=1)

mueller_rotate = RawRandomMuellerRotation(degrees=180, p=float('inf'))

from natsort import natsorted
dir_list = natsorted([str(el) for el in base_dir.iterdir() if el.is_dir() and not str(el).__contains__('C_2')])

for i, dir in enumerate(dir_list):
    intensity = read_cod_data_X3D(Path(dir) / 'raw_data' / '550nm' / '550_Intensite.cod', raw_flag=True)
    bruit = read_cod_data_X3D(Path(dir) / 'raw_data' / '550nm' /'550_Bruit.cod', raw_flag=True)
    B = (intensity - bruit).moveaxis(-1, 0)
    F = torch.cat([B, A.moveaxis(-1, 0), W.moveaxis(-1, 0)], dim=0)
    
    t = transforms[i]
    angle = angle + float(t[0]) if i > 0 else 0
    mueller_rotate.center = t[1:].tolist()
    F = mueller_rotate(F, angle=angle, transpose=True)
    y = mm_model(F[None])

    if i > 0:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(15, 8))
        axs[0].imshow(y_ref.squeeze().numpy())
        axs[0].set_title('Reference')
        axs[1].imshow(y_previous.squeeze().numpy())
        axs[1].set_title('Previous')
        axs[2].imshow(y.squeeze().numpy())
        axs[2].set_title('Current')
        plt.show()
    else:
        y_ref = y.clone()

    y_previous = y.clone()