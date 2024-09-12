import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage

def plot_fiber(raw_azimuth, linr, mask=None, intensity=None, n=10, option='quiver'):

	azimuth = np.pi*raw_azimuth/180
	X, Y = np.meshgrid(np.arange(azimuth.shape[1]), np.arange(azimuth.shape[0]))

	from skimage.restoration import unwrap_phase
	azimuth_unwrapped = unwrap_phase(azimuth-np.pi)#/2
	orientation_cos = np.cos(azimuth_unwrapped)
	orientation_sin = np.sin(azimuth_unwrapped)

	window = 5
	cos_mean = ndimage.uniform_filter(orientation_cos, (window, window))/window**2
	sin_mean = ndimage.uniform_filter(orientation_sin, (window, window))/window**2
	magnitude = (cos_mean**2 + sin_mean**2)**.5

	cos_mean_masked = np.ma.array(orientation_cos, mask=mask)
	sin_mean_masked = np.ma.array(orientation_sin, mask=mask)
	u = magnitude*linr*cos_mean_masked
	v = magnitude*linr*sin_mean_masked

	if mask is not None: u[mask] = 0
	if mask is not None: v[mask] = 0

	fig, ax = plt.subplots()
	if intensity is not None: ax.imshow(intensity, cmap = 'gray')

	if option == 'quiver':
		tracts = ax.quiver(
			X[::n, ::n], Y[::n, ::n], u[::n, ::n], v[::n, ::n],
			raw_azimuth[::n, ::n],
			scale = 20, 
			headwidth = 2, 
			cmap = plt.cm.twilight_shifted,
		)
		fig.colorbar(tracts, orientation='vertical')

	if option == 'streamplot':
		u = +cos_mean_masked
		v = -sin_mean_masked
		tracts = ax.streamplot(
			X, Y, u, v, color=azimuth_unwrapped, 
			density = 2, 
			cmap = plt.cm.twilight_shifted, 
			#norm = normalizer, 
			arrowsize = 0, 
			linewidth = 0.1*linr, 
			maxlength = 20., 
		)

	ax.axis('off')
	plt.tight_layout()
	from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
	canvas = FigureCanvas(fig)
	canvas.draw()

    # numpy array conversion
	w, h = fig.get_size_inches() * fig.get_dpi()
	quiver_img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(h), int(w), 3)

	return quiver_img
