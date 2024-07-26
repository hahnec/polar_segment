import time
import cv2
import scipy
import numpy as np
import ctypes


def removeReflections3D(I3D, maxThr=65530, SE=None, SErad=4):
    '''# Function to Remove Intensity-based supra-threshold specular reflections acquired with the polarimetric camera.
    # This is used to replace invalid pixels, whose intensity is saturated, in order to avoid spurious reflections.
    # This function is meant for Intensity polarimetric scans, assuming the input data is a 3D stack of Components.
    # this function embeds also a windowing filtering for integrating the replaced Intensity values in a smooth manner.
    #
    # Call: I3Drr = removeReflections3D( I3D , [maxThr], [SE], [SErad] )
    #
    # *Inputs*
	# I3D: 3D stack of Intensity Components of shape shp3 = [dim[0],dim[1],16]. These may have saturated pixels.
	#
	# [maxThr]: optional real value with the Camera max value for saturation (default: 65530).
	#
	# [SE]: structuring element for morphological operations (dilation of invalid pixels) (default: None -> Circular)
	#
	# [SErad]: radius in pixels for the automatic circular structuring element (default: 4)
	#
	# NB: the structuring element SE can be parsed as user-defined variable. It must be: logical 2D
	#
	# * Outputs *
	# I3Drr: 3D stack of Intensity Components with Removed Reflections.
	'''

    Isatmsk = np.any(I3D >= maxThr,axis=-1)

    if SE == None:
        SE = _getCircStrEl(SErad) # Create a circular structuring element for Dilation

    Isatmskdil = cv2.dilate(Isatmsk.astype(np.uint8), SE.astype(np.uint8)).astype(np.bool_)
    h2 = _getGaussWin2D(2 * SErad + 1)
    Iweight = cv2.filter2D((~Isatmskdil).astype(np.double), -1, h2)

    I3Dnan = np.array(I3D)
    I3Dnan[tile_Img2DtoImg3D(Isatmskdil)] = np.nan
    I3Dfix = fixNaNs3D(I3Dnan, verboseFlag=0)

    I3Drr = (I3D * np.sqrt(np.abs(tile_Img2DtoImg3D(Iweight)))) + \
            (I3Dfix * (1-np.sqrt(np.abs(tile_Img2DtoImg3D(Iweight)))))

    return I3Drr, ~Isatmskdil


def removeReflections2D(I2D, maxThr=65530, SE=None, SErad=4):
    '''# Function to Remove Intensity-based supra-threshold specular reflections acquired with the polarimetric camera.
    # This is used to replace invalid pixels, whose intensity is saturated, in order to avoid spurious reflections.
    # This function is meant for 2D Intensity polarimetric scans, assuming the input as a 2D individual image.
    # this function embeds also a windowing filtering for integrating the replaced Intensity values in a smooth manner.
    #
    # Call: I2Drr = removeReflections3D( I2D , [maxThr], [SE], [SErad] )
    #
    # *Inputs*
    # I2D: 2D image of an Intensity Component of shape shp2 = [dim[0],dim[1]]. These may have saturated pixels.
    #
    # [maxThr]: optional real value with the Camera max value for saturation (default: 65530).
    #
    # [SE]: structuring element for morphological operations (dilation of invalid pixels) (default: None -> Circular)
    #
    # [SErad]: radius in pixels for the automatic circular structuring element (default: 4)
    #
    # NB: the structuring element SE can be parsed as user-defined variable. It must be: logical 2D
    #
    # * Outputs *
    # I2Drr: 2D image of an Intensity Component with Removed Reflections.
    '''

    Isatmsk = I2D >= maxThr

    if SE == None:
        SE = _getCircStrEl(SErad) # Create a circular structuring element for Dilation

    Isatmskdil = cv2.dilate(Isatmsk.astype(np.uint8), SE.astype(np.uint8)).astype(np.bool_)
    h2 = _getGaussWin2D(2*SErad+1)
    Iweight = cv2.filter2D((~Isatmskdil).astype(np.double), -1, h2)

    I2Dnan = np.array(I2D)
    I2Dnan[Isatmskdil] = np.nan
    I2Dfix = fixNaNs2D(I2Dnan, verboseFlag=False)

    I2Drr = (I2D * np.sqrt(np.abs(Iweight))) + \
            (I2Dfix * (1 - np.sqrt(np.abs(Iweight))))

    return I2Drr, ~Isatmskdil


def _getCircStrEl(SErad=4):
    '''# Function to determine a 2D logical structuring element for morphological operations.
    # This function generates automatically a circular structuring element.
    #
    # Call: SE = _getCircStrEl( SErad )
    #
    # *Inputs*
    # [SErad]: radius in pixels for the automatic circular structuring element (default: 4)
    #
    # * Outputs *
    # SE: 2D logical circular structuring element.
    '''

    # Create a Circular Structuring Element for Morphological Operations
    xG, yG = np.meshgrid(
        np.linspace(-SErad, SErad, 2 * SErad + 1),
        np.linspace(-SErad, SErad, 2 * SErad + 1), indexing='ij')
    SE = np.sqrt(xG ** 2 + yG ** 2) <= SErad

    return SE


def _getGaussWin1D(wlen):
    '''# 1D Gaussian window: sigma = 1, shape [1, wlen], with Gain = 1.
    '''

    g1D = scipy.signal.windows.gaussian(wlen, 1).reshape((1, wlen))
    g1D = g1D / np.sum(g1D)
    return g1D


def _getGaussWin2D(wlen):
    '''# 2D Gaussian filter of shape [wlen, wlen], with Gain = 1.
    '''
    g1D = _getGaussWin1D(wlen)
    g2D = np.kron(g1D, np.transpose(g1D))
    g2D = g2D / np.sum(g2D)
    return g2D


def tile_Img2DtoImg3D(X2D, dim2=16):
    '''# Function to Replicate (tile) a 2D Image Component along the 3rd dimension, resulting into a 3D stack.
	#
	# Call: X3D = tile_Img2DtoImg3D(X2D,[dim2])
	#
	# *Inputs*
	# X2D: 2D Component of shape shp: [dim[0],dim[1]]
	#
	# [dim2]: optional integer scalar indicating the repetitions in the 3rd dimension
	#
	# *Outputs*
	# X3D: 3D stack of 2D Components of shape shp3.
	# 	   The matrix X3D will have shape equal to [dim[0],dim[1],[dim2=16]] by default.
	#
	'''

    X3D = np.tile(X2D.reshape([X2D.shape[0], X2D.shape[1], 1]), [1, 1, int(dim2)])

    return X3D


def fixNaNs3D(X3Dnan, verboseFlag=True):
    '''# Function to fix (correct) and replace NaN values in a 3D array (image as stack of 2D components).
     # This is used to fix invalid pixels, whose value is set to NaN, in order to fill missing data.
    # This function is a wrapper for a C-compiled code, and builds on a filling scheme based on the Euclidean distance
    # from a binary mask determined by the NaN values.
    # Although the Euclidean distance filling scheme is not optimal, such scheme allows for real-time performance
    # even with large patches of missing data (NaNs).
    # The function works for any real-valued 3D input data containing NaNs
    # NB: the location of NaNs must be consistent along the last dimension!.
    #
    # Call: X3Dfix = fixNaNs3D( X3Dnan, [verboseFlag] )
    #
    # *Inputs*
    # X3Dnan: 3D array (image as stack of 2D components) of real-values (double) of shape shp2 = [dim[0],dim[1],dim[2]].
    #         These may have NaN values, consistently along the last dimension.
    #         i.e.  if X3Dnan[x,y,0] == NaN     ->      it is assumed that:  X3Dnan[x,y,z] = NaN
    #
    # [verboseFlag]: scalar logical flag to enable verbose performance evaluation (default: 1)
    #
    # * Outputs *
    # X3Dfix: 3D array (image as stack of 2D components) of real-values (double) of shape shp2 = [dim[0],dim[1]],
    #         with new real-values in the correspondence of NaNs.
    #
    # NB: if all X3Dnan is NaN, no correction will be performed.
    '''

    if torch.isnan(X3Dnan).all():
        X3Dfix = X3Dnan
    else:
        X3DRnan = X3Dnan.clone()
        
        dims3 = torch.tensor(X3DRnan.shape, dtype=torch.float64)
        dims2 = torch.tensor([dims3[0], dims3[1]], dtype=torch.float64)
        X2DnanMsk = torch.isnan(X3DRnan[:, :, 0])
        X3Dfix = X3DRnan.clone().double()
        
        X2Dweight = torch.tensor(scipy.ndimage.distance_transform_edt(X2DnanMsk.numpy()), dtype=torch.float64)
        
        subR, subC = torch.where(X2DnanMsk)
        idxList = torch.tensor(np.ravel_multi_index([subR.numpy(), subC.numpy()], dims2.int().numpy()), dtype=torch.float64)
        lenList = torch.tensor(torch.sum(X2DnanMsk), dtype=torch.int32)
        
        if verboseFlag:
            t = time.time()
            for idx2 in idxList:
                fixVals_updateValues3D(X3Dfix, dims3, idx2)
            telaps = time.time() - t
            print(' >> fixNaNs: Elapsed time = {:.3f} s'.format(telaps))
        else:
            for idx2 in idxList:
                fixVals_updateValues3D(X3Dfix, dims3, idx2)

    return X3Dfix


def fixVals_updateValues3D(img3D, dims3, idx2):
    dims2 = [dims3[0], dims3[1]]
    r, c = ind2subs2D(idx2, dims2)
    d = math.sqrt(2)
    nnWgt = torch.tensor([d, 1.0, d, 1.0, 0.0, 1.0, d, 1.0, d], dtype=torch.float64)
    
    acc = 0.0
    val = torch.zeros(int(dims3[2]), dtype=torch.float64)
    
    lin3x3 = 9
    nr = torch.tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1])
    nc = torch.tensor([-1, 0, 1, -1, 0, 1, -1, 0, 1])
    
    for l in range(lin3x3):
        rr = r + nr[l]
        cc = c + nc[l]
        if 0 <= rr < dims2[0] and 0 <= cc < dims2[1]:
            for pp in range(int(dims3[2])):
                nn3idx = subs2ind3D(rr, cc, pp, dims3)
                if not torch.isinf(img3D[nn3idx]) and not torch.isnan(img3D[nn3idx]):
                    val[pp] += img3D[nn3idx] * nnWgt[l]
                    if pp == 0:
                        acc += nnWgt[l]
    
    for pp in range(int(dims3[2])):
        idx3 = subs2ind3D(r, c, pp, dims3)
        img3D[idx3] = val[pp] / acc


import torch
import math

def ind2subs2D(idx, dims):
    r = int(idx // dims[1])
    c = int(idx % dims[1])
    return r, c

def subs2ind3D(r, c, p, dims):
    return int(r * dims[1] * dims[2] + c * dims[2] + p)

def fixVals_updateValues3D(img3D, dims3, idx2):
    dims2 = [dims3[0], dims3[1]]
    r, c = ind2subs2D(idx2, dims2)
    d = math.sqrt(2)
    nnWgt = torch.tensor([d, 1.0, d, 1.0, 0.0, 1.0, d, 1.0, d], dtype=torch.float64)
    
    acc = 0.0
    val = torch.zeros(int(dims3[2]), dtype=torch.float64)
    
    lin3x3 = 9
    nr = torch.tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1])
    nc = torch.tensor([-1, 0, 1, -1, 0, 1, -1, 0, 1])
    
    for l in range(lin3x3):
        rr = r + nr[l]
        cc = c + nc[l]
        if 0 <= rr < dims2[0] and 0 <= cc < dims2[1]:
            for pp in range(int(dims3[2])):
                nn3idx = subs2ind3D(rr, cc, pp, dims3)
                if not torch.isinf(img3D[nn3idx]) and not torch.isnan(img3D[nn3idx]):
                    val[pp] += img3D[nn3idx] * nnWgt[l]
                    if pp == 0:
                        acc += nnWgt[l]
    
    for pp in range(int(dims3[2])):
        idx3 = subs2ind3D(r, c, pp, dims3)
        img3D[idx3] = val[pp] / acc

# Example usage:
X3Dnan = torch.tensor(np.random.random((5, 5, 3)), dtype=torch.float64)
X3Dnan[2, 2, :] = float('nan')

fixed_X3D = fixNaNs3D(X3Dnan, verboseFlag=True)
print(fixed_X3D)