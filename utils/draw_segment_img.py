import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import draw_segmentation_masks

from mm.utils.draw_fiber_img import plot_fiber


def draw_segment_maps(imgs, preds, bg=None, bidx=0):

    # extract intensity images for plots
    bg_opt = True if bg is not None and isinstance(bg, torch.Tensor) else False
    frame_pred = draw_segmentation_img(imgs, preds, bidx=bidx, bg_opt=bg_opt)
    if bg_opt:
        alpha = (~bg[bidx]).float()*255
        frame_pred = torch.cat((frame_pred, alpha), dim=0)

    return frame_pred

def draw_heat_maps(imgs, preds, bg=None, bidx=0, out_class=-1):

    # extract intensity images for plots
    bg_opt = True if bg is not None and isinstance(bg, torch.Tensor) else False
    hmask = (preds[bidx].argmax(0) == 0) if bg_opt else None
    heatmap = draw_heatmap(preds[bidx, out_class], img=imgs[bidx], mask=hmask)
    if bg_opt:
        alpha = (~bg[bidx]).float()*255
        heatmap = np.concatenate((heatmap, (~bg[bidx]).float().moveaxis(0, -1).cpu().numpy()), axis=-1)

    return heatmap

def draw_fiber_maps(raw_frame, preds, azimuth_model, bg=None, bidx=0):

    bg_opt = True if bg is not None and isinstance(bg, torch.Tensor) else False

    # fiber tracts image
    lc_feats = azimuth_model(raw_frame)
    imgs = raw_frame.mean(1)
    masks = (preds.argmax(1) == 0) | (preds.argmax(1) == 1) # predicted healthy and tumor white matter mask
    feats = [var[bidx].cpu().numpy() for var in [lc_feats, masks, imgs]]
    mask = ~(feats[1] & ~bg[bidx, 0].numpy()) if bg_opt else ~feats[1]
    # fiber plot
    azi, linr, img = feats[0][1], feats[0][0], feats[2]
    fiber_img, cbar_img = plot_fiber(raw_azimuth=azi, linr=10, intensity=img, mask=mask)
    azi_img = plt.cm.twilight_shifted(azi/180)
    azi_img = ((azi_img-azi_img.min())/(azi_img.max()-azi_img.min()) * 255).astype(np.uint8)    # uint8 norm
    if bg_opt:
        alpha = (~bg[bidx]).float()*255
        fiber_img = np.concatenate((fiber_img, alpha.permute(1,2,0).numpy()), axis=-1)
        azi_img = np.concatenate((azi_img[..., :3], alpha.permute(1,2,0).numpy()), axis=-1)

    return fiber_img, azi_img, cbar_img

def draw_segmentation_imgs(imgs, preds, truth, bidx=0, bg_opt=False, alpha=0.3):

    n_channels = preds.shape[1]
    colors = ['yellow', 'green', 'red', 'blue'] if n_channels-int(bg_opt) > 2 else ['yellow', 'green', 'red'] 
    preds_b = torch.nn.functional.one_hot(preds.argmax(1), num_classes=n_channels).permute(0, 3, 1, 2).bool()
    combined_masks = torch.stack((preds_b[bidx], truth[bidx]>0)).cpu()
    combined_masks[0][:, combined_masks[1].sum(0)==0] = 0  # remove predictions for areas where there is no GT
    img = (imgs[bidx][None].repeat(3, 1, 1)/imgs[bidx].max()*255).cpu().to(torch.uint8)
    frame_pred = draw_segmentation_masks(img, masks=combined_masks[0], alpha=alpha, colors=colors[-n_channels:])
    frame_mask = draw_segmentation_masks(img, masks=combined_masks[1], alpha=alpha, colors=colors[-n_channels:])

    return frame_pred, frame_mask

def draw_segmentation_img(imgs, preds, bidx=0, bg_opt=False, alpha=0.3):

    n_channels = preds.shape[1]
    colors = ['yellow', 'green', 'red', 'blue'] if n_channels-int(bg_opt) > 2 else ['yellow', 'green', 'red'] 
    preds_b = torch.nn.functional.one_hot(preds.argmax(1), num_classes=n_channels).permute(0, 3, 1, 2).bool()
    combined_masks = preds_b[bidx].cpu()
    img = (imgs[bidx][None].repeat(3, 1, 1)/imgs[bidx].max()*255).cpu().to(torch.uint8)
    frame_pred = draw_segmentation_masks(img, masks=combined_masks, alpha=alpha, colors=colors[-n_channels:])

    return frame_pred

def draw_heatmap(pred, img=None, mask=None, alpha=0.3, colormap='jet'):

    if isinstance(pred, torch.Tensor): pred = pred.detach().cpu().numpy()
    norm = (pred - pred.min()) / (pred.max() - pred.min())

    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(colormap)
    heatmap = cmap(norm)

    if mask is not None:
        if isinstance(mask, torch.Tensor): mask = mask.bool().cpu().numpy()
        heatmap[mask] = 0

    if img is not None:
        if isinstance(img, torch.Tensor): img = img.cpu().numpy()
        if len(img.shape) == 2: img = img[..., None]
        img = (img - img.min()) / (img.max() - img.min())
        heatmap = img * (1 - alpha) + heatmap[..., :3] * alpha
    
    return heatmap


def create_color_bar(colormap='jet', orientation='horizontal', output_file='color_bar.svg'):
    """
    Plots a color bar using a specified colormap and saves it as an SVG file with a tight bounding box.

    Args:
        colormap_name (str): Name of the colormap to use.
        output_file (str): Name of the output SVG file.
    """

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # Create a figure and axis for the color bar
    fig, ax = plt.subplots(figsize=(6, 1), dpi=100)

    # Create a colormap and normalize from 0 to 1
    cmap = plt.get_cmap(colormap)
    norm = mcolors.Normalize(vmin=0, vmax=1)

    # Create a color bar
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation=orientation)

    # Set color bar labels
    cb.set_ticks([0, 1])
    cb.set_ticklabels(['0', '1'])
    
    # Set font size for the tick labels
    cb.ax.tick_params(labelsize=25)

    # Save the color bar to a file with a tight bounding box
    fig.savefig(output_file, bbox_inches='tight', format='svg', transparent=True)
    plt.close(fig)


def create_legend(labels, colors, legend_size, alpha=0.3):

    import matplotlib.pyplot as plt
    # Create a figure and axis for the legend
    fig, ax = plt.subplots(figsize=(legend_size[1] / 100, legend_size[0] / 100), dpi=100)
    handles = [plt.Line2D([0], [0], color=color, lw=6, alpha=alpha) for color in colors]
    ax.legend(handles, labels, loc='center', frameon=False, fontsize=10)
    ax.axis('off')

    # Save the legend to a file with a transparent background
    fig.savefig('legend.svg', bbox_inches='tight', pad_inches=0, transparent=True)
    fig.savefig('legend.png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)


def overlay_legend(frame, labels, colors, alpha=0.3, legend_position=(10, 10), legend_size=(100, 300)):
    """
    Overlays a transparent legend onto a given image tensor.

    Args:
        frame (torch.Tensor): The image tensor (C, H, W) to overlay the legend on.
        labels (list of str): List of labels for the legend.
        colors (list of str): List of colors corresponding to the labels.
        legend_position (tuple): Top-left position (x, y) where the legend will be placed.
        legend_size (tuple): Size (width, height) of the legend.

    Returns:
        torch.Tensor: The image tensor with the transparent legend overlaid.
    """

    import numpy as np
    from PIL import Image

    create_legend(labels, colors, legend_size, alpha=alpha)

    # Load the legend image
    legend_image = Image.open('legend.png').convert('RGBA')

    # Resize the legend image to the specified size
    legend_image = legend_image.resize(legend_size) #, Image.ANTIALIAS
    legend_image_np = np.array(legend_image)
    
    # Convert the legend image to a tensor
    legend_image_tensor = torch.tensor(legend_image_np).permute(2, 0, 1).float() / 255.0

    # Define the position to overlay the legend on the image
    overlay_height, overlay_width = legend_size[1], legend_size[0]
    overlay_y, overlay_x = legend_position
    image_with_legend = frame.clone()

    # Ensure the overlay fits within the main image
    if overlay_y + overlay_height > image_with_legend.shape[1] or overlay_x + overlay_width > image_with_legend.shape[2]:
        raise ValueError("The legend %s overlay exceeds the dimensions of the main image." % str(image_with_legend.shape))

    # Apply alpha blending to combine the legend with the main image
    for c in range(3):  # Assuming 3 channels (RGB)
        image_with_legend[c, overlay_y:overlay_y+overlay_height, overlay_x:overlay_x+overlay_width] = (
            image_with_legend[c, overlay_y:overlay_y+overlay_height, overlay_x:overlay_x+overlay_width] * (1 - legend_image_tensor[3]) +
            legend_image_tensor[c] * legend_image_tensor[3]
        )

    return image_with_legend
