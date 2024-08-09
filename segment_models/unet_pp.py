import torch
import torch.nn as nn
import torch.nn.functional as F


def get_pretrained_unet_pp(n_channels, out_channels=2):

    from transformers import AutoModelForImageSegmentation
    model = AutoModelForImageSegmentation.from_pretrained("voitl/unet_plus_plus", trust_remote_code=True)

    # adapt input channels
    conv1 = model.model.encoder.conv1
    conv1_weight_shape = conv1.weight.shape
    new_weight_shape = (conv1_weight_shape[0], n_channels) + conv1_weight_shape[2:]
    new_weight = torch.zeros(new_weight_shape)
    with torch.no_grad():
        if n_channels == 1:
            new_weight.copy_(conv1.weight)
        elif n_channels > 1:
            for i in range(n_channels):
                new_weight[:, i:i+1] = conv1.weight / n_channels
        else:
            raise ValueError("n_channels must be a positive integer")
    conv1.weight = nn.Parameter(new_weight)

    # adapt output channels
    if out_channels != 16: 
        model.model.segmentation_head[0] = nn.Conv2d(64, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        nn.init.kaiming_normal_(model.model.segmentation_head[0].weight, mode='fan_out', nonlinearity='relu')
        
    return model


class UnetPP(torch.nn.Module):
    def __init__(self, n_channels, out_channels=2):
        super(UnetPP, self).__init__()

        self.parent_model = get_pretrained_unet_pp(n_channels, out_channels)

    def forward(self, x):
        # Original height and width
        orig_height, orig_width = x.shape[2], x.shape[3]
        
        # Calculate padding
        pad_height = (32 - orig_height % 32) % 32
        pad_width = (32 - orig_width % 32) % 32
        
        # Apply padding
        x = F.pad(x, (0, pad_width, 0, pad_height), mode='constant', value=0)
        
        # Forward pass through the parent model
        x = self.parent_model(x)
        
        # Crop the output back to the original dimensions
        x = x[:, :, :orig_height, :orig_width]
        
        return x
