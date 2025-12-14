import numpy as np
import matplotlib.pyplot as plt

def make_anaglyph(left, right, normalize=True, show=True, save_fpath=None, figsize=(8,6), ax=None):
    """
    Create a red-cyan anaglyph from `left` and `right` images.

    Parameters:
    - left, right: np.ndarray images with same shape (H,W) or (H,W,3).
      If single-channel (H,W), treated as luminance.
    - normalize: if True convert float images to 0-255 uint8; if images are float in [0,1] they are rescaled.
    - show: if True display the anaglyph with matplotlib; otherwise return the RGB array.
    - save_fpath: if not None, save anaglyph as provided file name
    - figsize: figure size passed to plt.figure when show=True.
    - ax: optional matplotlib Axes to draw into when show=True.

    Returns:
    - anaglyph: (H,W,3) uint8 RGB image (left->red, right->cyan).
    """
    if left.shape[:2] != right.shape[:2]:
        raise ValueError("left and right must have same height and width")

    # Convert grayscale to RGB by stacking
    def to_rgb(img):
        if img.ndim == 2:
            return np.stack([img]*3, axis=-1)
        if img.ndim == 3 and img.shape[2] == 3:
            return img
        raise ValueError("Input images must be HxW or HxWx3 arrays")

    L = to_rgb(np.array(left))
    R = to_rgb(np.array(right))

    # Convert to float in 0..1 for blending calculations if needed
    orig_dtype = L.dtype
    if np.issubdtype(L.dtype, np.floating) or np.issubdtype(R.dtype, np.floating):
        Lf = L.astype(np.float32)
        Rf = R.astype(np.float32)
        if normalize:
            # if floats appear in 0..1 assume that range, otherwise scale by max
            if Lf.max() <= 1.0: Lf = np.clip(Lf, 0.0, 1.0)
            else: Lf = Lf / max(1.0, Lf.max())
            if Rf.max() <= 1.0: Rf = np.clip(Rf, 0.0, 1.0)
            else: Rf = Rf / max(1.0, Rf.max())
    else:
        # ints: convert to 0..1 float
        Lf = L.astype(np.float32) / 255.0
        Rf = R.astype(np.float32) / 255.0

    # Compose anaglyph:
    # - red channel from left image
    # - green and blue (cyan) channels from right image
    A = np.zeros_like(Lf)
    A[..., 0] = Lf[..., 0]  # red from left
    A[..., 1] = Rf[..., 1]  # green from right
    A[..., 2] = Rf[..., 2]  # blue from right

    # If inputs were grayscale (stacked identical channels), consider averaging channels for nicer result:
    # detect near-equal channels and replace with luminance
    def is_gray(img):
        return np.allclose(img[...,0], img[...,1]) and np.allclose(img[...,0], img[...,2])
    if is_gray(Lf):
        Llum = np.clip(0.299*Lf[...,0] + 0.587*Lf[...,1] + 0.114*Lf[...,2], 0, 1)
        A[...,0] = Llum
    if is_gray(Rf):
        Rlum = np.clip(0.299*Rf[...,0] + 0.587*Rf[...,1] + 0.114*Rf[...,2], 0, 1)
        A[...,1] = Rlum
        A[...,2] = Rlum

    # convert back to uint8 0..255
    An = np.clip((A * 255.0).round(), 0, 255).astype(np.uint8)

    if show or save_fpath is not None:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(An)
        ax.axis('off')
        if show and ax is None:
            plt.show()
        if save_fpath is not None:
            plt.savefig(save_fpath)
        return An
    return An
