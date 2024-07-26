import cv2
import scipy
import numpy as np


def specular_removal_cv(img, th=65530, method='navier', size=3):

    chs = img.shape[-1]
    result = np.zeros_like(img)
    mask = (img > th).astype(np.uint8)
    for ch in range(chs):
        # Perform morphological dilation
        if False:
            smask = scipy.ndimage.maximum_filter(mask[..., ch], size=size, mode='constant')
        elif False:
            structure_kernel = np.ones((size, size), dtype=np.uint8)
            structure_kernel[1, 1] = 0
            structure_kernel[0, 0] = 2**.5
            structure_kernel[2, 2] = 2**.5
            structure_kernel[0, 2] = 2**.5
            structure_kernel[2, 0] = 2**.5
            smask = scipy.ndimage.binary_dilation(mask[..., ch], structure=structure_kernel)
        elif False:
            smask = scipy.ndimage.distance_transform_edt(np.array(mask[..., ch]))
        elif True:
            #from libmpMuelMat_dev_1_0.libmpMuelMat import _getGaussWin2D, tile_Img2DtoImg3D
            xG, yG = np.meshgrid(
                np.linspace(-size, size, 2 * size + 1),
                np.linspace(-size, size, 2 * size + 1), indexing='ij'
            )
            SE = np.sqrt(xG ** 2 + yG ** 2) <= size
            smask = cv2.dilate(mask[..., ch], SE.astype(np.uint8)).astype(np.bool_)
            #h2 = _getGaussWin2D(2 * size + 1)
            #Iweight = cv2.filter2D((~smask).astype(np.double), -1, h2)

            smask = (smask * 255).astype(np.uint8)

            if method == 'telea':
                result[..., ch] = cv2.inpaint(img[..., ch][..., None], smask[..., None], size, cv2.INPAINT_TELEA)
            elif method == 'navier':
                result[..., ch] = cv2.inpaint(img[..., ch][..., None], smask[..., None], size, cv2.INPAINT_NS)
            else:
                NotImplementedError('Method %s not recognized' % method)

        #result = (img * np.sqrt(np.abs(tile_Img2DtoImg3D(Iweight)))) + (result * (1-np.sqrt(np.abs(tile_Img2DtoImg3D(Iweight)))))

    return result, smask
