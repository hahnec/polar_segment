from pathlib import Path
import pickle
from skimage import transform, filters
from scipy.optimize import leastsq

base_dir = Path('/media/chris/EB62-383C/CC_Rotation/')

object_fun = lambda deg, px, py, x, ref: (transform.rotate(x, deg, center=(px, py), mode='edge')[:, c:-c] - ref).flatten()**2

from natsort import natsorted
dir_list = natsorted([str(el) for el in base_dir.iterdir()])

for i, dir in enumerate(dir_list):
    rimg_path = Path(dir) / 'polarimetry' / '550nm' / 'MM.pickle'
    with open(rimg_path, 'rb') as f:
        rimg = pickle.load(f)['nM'].mean(-1)
    if i == 0:
        c = (rimg.shape[-1]-rimg.shape[0])//2
        ref = rimg[:, c:-c]
        continue
    
    xfun = lambda arg: object_fun(arg[0], arg[1], arg[2], x=filters.gaussian(rimg, sigma=2), ref=filters.gaussian(ref, sigma=2))
    p = leastsq(xfun, [i*10., rimg.shape[1]/2, rimg.shape[0]/2])[0]
    print(p)

    new = transform.rotate(rimg, p[0], center=(p[1], p[2]), mode='edge')[:, c:-c]

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(18, 7))
    axs[0].set_title('Reference')
    axs[1].set_title('Rotated by %s°' % str(round(p[0], 2)))
    axs[2].set_title('Difference')
    axs[0].imshow(ref)
    axs[1].imshow(new)
    axs[2].imshow(abs(ref-new))
    plt.show()
