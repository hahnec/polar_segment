# %%

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
#import cmocean
from skimage.restoration import unwrap_phase
import scipy.ndimage as ndimage

plt.rcParams["figure.figsize"] = (20,10)

# %%
polarimetry_case = '/data/disk3/Polarimetry/Fixed_samples/2021-10-26_T_B3S1AUT_FX_M_7/650nm/'
polarimetry_case = '/home/chris/Datasets/03_HORAO/TumorMeasurementsCalib/batch1/64/2022-12-06_T_HORAO-64-AF_FR_S1_1/polarimetry/550nm/'

# %%
#from scipy.io import loadmat
#orientation = loadmat(polarimetry_case+'M11.mat')
import numpy as np
orientation = np.load(polarimetry_case+'MM.npz')

# %%
M11 = orientation['M11']
#depolarization = orientation['Image_total_depolarization']
print(dict(orientation).keys())
depolarization = orientation['totP']
image_size = M11.shape

lower_M11 = np.mean(M11)-2*np.std(M11)
upper_M11 = np.mean(M11)+2*np.std(M11)

# %%
#azimuth = np.pi*orientation['Image_orientation_linear_retardance_full']/360
azimuth = np.pi*orientation['azimuth']/180

# %%
#azimuth_unwrapped = unwrap_phase(4*azimuth)/2
azimuth_unwrapped = unwrap_phase(2*azimuth)/2
azimuth_unwrapped = unwrap_phase(azimuth-np.pi)#/2


# %%
cycle_limits = (np.floor(np.min(azimuth_unwrapped/(np.pi))), np.ceil(np.max(azimuth_unwrapped/(np.pi))))

# %%
num_cycles = cycle_limits[1]-cycle_limits[0]

# %%
cut_points = np.linspace(0,1.0, int(num_cycles+1))
intervals = zip(cut_points[:-1],cut_points[1:])

# %%
#cmaps = [list(zip(np.linspace(x,y,128),cmocean.cm.phase(np.linspace(0,1.,128)))) for x,y in zip(cut_points[:-1],cut_points[1:])]
cmaps = [list(zip(np.linspace(x, y, 128), plt.cm.twilight_shifted(np.linspace(0, 1, 128)))) 
         for x, y in zip(cut_points[:-1], cut_points[1:])]


# %%
new_cmap_values = [x for y in cmaps for x in y[:-1]]
new_cmap_values.append(cmaps[-1][-1])

interval_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('interval_cmap',new_cmap_values)


# %%
plt.rcParams["figure.figsize"] = (10,6)

# %%
plt.imshow(M11, cmap='gray', vmin = lower_M11, vmax = upper_M11)
cb =plt.colorbar()

cb.ax.tick_params(labelsize=16)
plt.savefig('intensity.eps', format='eps')

# %%
X,Y = np.meshgrid(np.arange(image_size[1]), np.arange(image_size[0]))

# %%
normalizer = matplotlib.colors.Normalize(vmin=cycle_limits[0]*np.pi, vmax=cycle_limits[1]*np.pi, clip=False)

# %%
orientation_cos = np.cos(azimuth_unwrapped)
orientation_sin = np.sin(azimuth_unwrapped)

window = 5

cos_mean = ndimage.uniform_filter(orientation_cos, (window, window))/(window*window)
sin_mean = ndimage.uniform_filter(orientation_sin, (window, window))/(window*window)

magnitude = np.sqrt(np.square(cos_mean) + np.square(sin_mean))


# %%
mask_from_unet = np.ones_like(azimuth)#np.load('2021-10-26_T_B3S1AUT_mask.npy')

# %%
n=10
cos_mean_masked = np.ma.array(orientation_cos, mask = mask_from_unet<0.95)
sin_mean_masked = np.ma.array(orientation_sin, mask=  mask_from_unet<0.95)
#u = (magnitude*orientation['Image_linear_retardance']*cos_mean_masked)[::n,::n]
#v= (magnitude*orientation['Image_linear_retardance']*sin_mean_masked)[::n,::n]
u = (magnitude*orientation['linR']*cos_mean_masked)[::n,::n]
v= (magnitude*orientation['linR']*sin_mean_masked)[::n,::n]
plt.imshow(M11, vmax = upper_M11, cmap = 'gray')
plt.quiver(X[::n,::n], 
           Y[::n,::n], 
           u, 
           v,
           #orientation['Image_orientation_linear_retardance_full'][::n,::n],
           orientation['azimuth'][::n,::n],
           scale = 20,
          headwidth=2, 
          #cmap=cmocean.cm.phase,
          cmap=plt.cm.twilight_shifted,
          )

cb =plt.colorbar()

cb.ax.tick_params(labelsize=12)

#plt.savefig('masked_quiver.eps', format='eps')
plt.show()

# %%
u = (cos_mean_masked)
v= -(sin_mean_masked)
plt.imshow(M11, vmin = lower_M11, vmax = upper_M11, cmap = 'gray')
plt.streamplot(X, Y, u, v,color=azimuth_unwrapped,
        density = 2,cmap=interval_cmap, norm = normalizer, arrowsize=0, 
        #linewidth = 0.1*orientation['Image_linear_retardance'],
        linewidth = 0.1*orientation['linR'],
              maxlength=20.)

# %%
u = (cos_mean_masked)
v= -(sin_mean_masked)
plt.imshow(M11, vmin = lower_M11, vmax = upper_M11, cmap = 'gray')
plt.streamplot(X, Y, u, v,color=azimuth_unwrapped,
          density = 2, norm = normalizer, arrowsize=0, 
          #linewidth = 0.1*orientation['Image_linear_retardance'],
          linewidth = 0.1*orientation['linR'],
              )

# %%



