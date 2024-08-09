import torch.nn as nn
import torch.nn.init as init

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.normal_(m.bias, std=1e-6)
    elif isinstance(m, nn.ConvTranspose2d):
        # Optionally initialize transposed conv layers for bilinear interpolation
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.normal_(m.bias, std=1e-6)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.normal_(m.bias, std=1e-6)
    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.normal_(m.bias, std=1e-6)
