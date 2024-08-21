from pathlib import Path
import pickle
from skimage import transform, filters
from scipy.optimize import leastsq

base_dir = Path('/media/chris/EB62-383C/CC_Rotation_/')
plot_opt = False
object_fun = lambda deg, px, py, x, ref: (transform.rotate(x, deg, center=(px, py), mode='edge')[s:-s, s+c:-s-c] - ref).flatten()**2

from natsort import natsorted
dir_list = natsorted([str(el) for el in base_dir.iterdir() if el.is_dir()])

for i, dir in enumerate(dir_list):
    rimg_path = Path(dir) / 'polarimetry' / '550nm' / 'MM.pickle'
    with open(rimg_path, 'rb') as f:
        rimg = pickle.load(f)['nM'].mean(-1)
    if i == 0:
        s = 50
        c = (rimg.shape[-1]-rimg.shape[0])//2
        ref = rimg[s:-s, s+c:-s-c]
        continue
    
    xfun = lambda arg: object_fun(arg[0], arg[1], arg[2], x=filters.gaussian(rimg, sigma=3), ref=filters.gaussian(ref, sigma=3))
    p = leastsq(xfun, [10., rimg.shape[1]/2, rimg.shape[0]/2])[0]
    print(p)

    with open(base_dir / 'angles_and_center_points.txt', 'a') as f:
        f.writelines(' '.join([str(el) for el in p])+'\n')

    new = transform.rotate(rimg, p[0], center=(p[1], p[2]), mode='edge')[s:-s, s+c:-s-c]

    if plot_opt:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(18, 7))
        axs[0].set_title('Reference')
        axs[1].set_title('Rotated by %s°' % str(round(p[0], 2)))
        axs[2].set_title('Difference')
        axs[0].imshow(filters.gaussian(ref, sigma=3))
        axs[1].imshow(filters.gaussian(new, sigma=3))
        axs[2].imshow(abs(filters.gaussian(ref, sigma=3)-filters.gaussian(new, sigma=3)))
        plt.show()

    # use last frame as reference frame
    ref = rimg[s:-s, s+c:-s-c]
