import torch
import torch.nn.functional as F
import scipy
import numpy as np


def specular_removal_t(frame, mask, size=3):
    
    chs = mask.shape[-1]
    mask = np.array(mask)
    #for ch in range(chs):
    #    mask[..., ch] = scipy.ndimage.distance_transform_edt(mask[..., ch])
    mask = torch.tensor(mask, device=frame.device)
    ffix = frame.clone()
    ffix[mask] = torch.nan
    #means = batched_rolling_window_metric(frame, patch_size=size, function=torch.nanmean)
    means = update_nan_values_pytorch(frame)
    #means = nanmean_sliding_window(ffix)
    #means = fixVals_updateValues3D(frame, dims3, idx2)
    means = means.unsqueeze(-1).repeat(1, 1, 16)

    h2 = get_gaussian_kernel(2 * size + 1, sigma=1.0)
    smask_double = (~mask[..., 0]).to(h2.dtype).unsqueeze(0).unsqueeze(0)
    h2 = h2.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    weights = F.conv2d(smask_double, h2, padding=h2.shape[-1] // 2).squeeze().unsqueeze(-1).repeat(1, 1, 16)

    #frame[mask] = means[mask]
    frame = (frame * np.sqrt(np.abs(weights))) + (means * (1-np.sqrt(np.abs(weights))))

    return frame


def update_nan_values_pytorch(img3D):
    '''
    Update NaN values in a 3D tensor using a 3x3 spatial kernel repeated across channels.
    Computes the nanmean of non-NaN values in the current 3D window.
    
    Parameters:
    img3D (torch.Tensor): Input 3D tensor with NaN values to be fixed (shape: (h, w, c))
    
    Returns:
    updated_img (torch.Tensor): Input tensor with NaN values updated
    '''
    # Define the 3x3 spatial kernel with the specified weights
    d = 2**.5
    kernel = torch.tensor([
        [[d, 1.0, d],
         [1.0, 0.0, 1.0],
         [d, 1.0, d]],
    ], dtype=img3D.dtype)

    # Normalize the kernel to sum to 1, ignoring the center
    kernel_sum = kernel.sum() - kernel[0, 1, 1]
    kernel = kernel / kernel_sum
    kernel[0, 1, 1] = 0.0

    # Repeat the 2D kernel across channels to create a 3D kernel
    kernel = kernel.permute(1,2,0).repeat(1, 1, 1, img3D.size(2))

    # Find NaN positions in the input tensor
    nan_mask = torch.isnan(img3D)

    # Perform 3D convolution to compute nanmean
    padded_img3D = F.pad(img3D.unsqueeze(0).unsqueeze(0), (0, 0, 1, 1, 1, 1), mode='constant', value=float('nan'))
    conv_result = F.conv3d(padded_img3D, kernel.unsqueeze(0), padding=0)

    # Calculate the nanmean across channels
    valid_mask = ~torch.isnan(img3D)
    nanmean_values = torch.sum(conv_result * valid_mask.unsqueeze(0).unsqueeze(0), dim=1) / torch.sum(valid_mask, dim=2, keepdim=True)

    # Replace NaN values in the input tensor with the computed nanmean values
    #img3D[nan_mask] = nanmean_values[nan_mask]

    return nanmean_values #img3D


def nanmean_sliding_window(img3D):
    '''
    Update NaN values in a 3D tensor using a sliding window approach with a 3x3x3 kernel.
    Computes the nanmean of non-NaN values in the current 3D window.
    
    Parameters:
    img3D (torch.Tensor): Input 3D tensor with NaN values to be fixed (shape: (h, w, c))
    
    Returns:
    updated_img (torch.Tensor): Input tensor with NaN values updated
    '''
    h, w, c = img3D.shape
    
    # Define the 3x3x3 spatial kernel with the specified weights
    d = 2**.5
    kernel = torch.tensor([
        [[d, 1.0, d],
         [1.0, 0.0, 1.0],
         [d, 1.0, d]],
    ], dtype=torch.float32)

    # Normalize the kernel to sum to 1, ignoring the center
    kernel_sum = kernel.sum() - kernel[0, 1, 1]
    kernel = kernel / kernel_sum
    kernel[0, 1, 1] = 0.0

    # Initialize output tensor with NaN values replaced by zeros
    updated_img = img3D.clone()
    updated_img[torch.isnan(updated_img)] = 0.0
    
    # Extract sliding windows using unfold
    #unfold_img = F.unfold(updated_img.unsqueeze(0), kernel_size=(3, 3, c), padding=1)
    unfold_img = F.unfold(updated_img.permute(2, 0, 1).unsqueeze(1), kernel_size=(3, 3), padding=1)
    #unfold_img = unfold_img.squeeze(0)
    
    mimg = unfold_img * kernel.flatten(1, 2).unsqueeze(-1)
    nanmean_values = mimg.nanmean(0).nanmean(0)  # average across spatial domain
    #nanmean_values = mimg.nanmean(0).nansum(0)
    nanmean_values = nanmean_values.view(h, w)
    
    # Replace NaN values in the original tensor with nanmean_values
    #nan_mask = torch.isnan(updated_img)
    #updated_img[nan_mask] = nanmean_values.view(h, w)[nan_mask]
    
    return nanmean_values #updated_img

import torch

# Ancillary Function for indexing (ind2subs2D) in 2D -- C-like ordering for indexing
def ind2subs2D(idx, dims):
    r = torch.zeros_like(idx)
    c = torch.zeros_like(idx)
    fracPart = idx / dims[1]
    r = fracPart.floor()
    c = (fracPart - r) * dims[1]
    return r, c

# Ancillary Function for indexing (ind2subs3D) in 3D -- C-like ordering for indexing
def ind2subs3D(idx, dims):
    r = torch.zeros_like(idx)
    c = torch.zeros_like(idx)
    p = torch.zeros_like(idx)
    
    fracPart = idx / (dims[1] * dims[2])
    r = fracPart.floor()
    fracPart = (fracPart - r) * dims[1]
    c = fracPart.floor()
    p = (fracPart - c) * dims[2]
    
    return r, c, p

# Ancillary Function for indexing (subs2ind2D) in 2D -- C-like ordering for indexing
def subs2ind2D(r, c, dims2):
    return c + r * dims2[1]

# Ancillary Function for indexing (subs2ind3D) in 3D -- C-like ordering for indexing
def subs2ind3D(r, c, p, dims3):
    return p + c * dims3[2] + r * (dims3[1] * dims3[2])

# Ancillary Function for sorting value-based indexes
def cmpDouble(values):
    sorted_indices = sorted(range(len(values)), key=lambda k: values[k]['value'])
    return sorted_indices


def fixVals_updateValues3D(img3D, dims3, idx2):
    dims2 = torch.tensor([dims3[0], dims3[1]], dtype=torch.double)
    r, c = ind2subs2D(idx2, dims2)
    
    d = 2**.5
    nnWgt = torch.tensor([d, 1.0, d, 1.0, 0.0, 1.0, d, 1.0, d], dtype=torch.double)
    
    acc = torch.tensor(0.0, dtype=torch.double)
    val = torch.zeros(int(dims3[2]), dtype=torch.double)
    
    lin3x3 = 9
    nr = torch.tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1])
    nc = torch.tensor([-1, 0, 1, -1, 0, 1, -1, 0, 1])
    
    for l in range(lin3x3):
        rr = r + nr[l]
        cc = c + nc[l]
        valid_mask = (rr >= 0) & (rr < dims2[0]) & (cc >= 0) & (cc < dims2[1])
        
        for pp in range(int(dims3[2])):
            if valid_mask:
                nn3idx = subs2ind3D(rr, cc, float(pp), dims3)
                if not torch.isinf(img3D[int(nn3idx)]) and not torch.isnan(img3D[int(nn3idx)]):
                    val[pp] += img3D[int(nn3idx)] * nnWgt[l]
                    if pp == 0:
                        acc += nnWgt[l]
    
    for pp in range(int(dims3[2])):
        idx3 = subs2ind3D(r, c, float(pp), dims3)
        img3D[int(idx3)] = val[pp] / acc if acc != 0 else 0.0
    
    return img3D


def get_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    """Generates a 2D Gaussian kernel."""
    x = torch.arange(kernel_size).float() - kernel_size // 2
    x = torch.exp(-x ** 2 / (2 * sigma ** 2))
    kernel_1d = x / x.sum()
    kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]
    return kernel_2d
