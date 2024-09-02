import torch
from torch import nn
from tqdm import tqdm

class PatchResNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, patch_size=50, testing=False):
        super(PatchResNet, self).__init__()

        from monai.networks.nets import resnet50
        self.model = resnet50(pretrained=False, spatial_dims=2, n_input_channels=n_channels, num_classes=n_classes)
        self.patch_size = patch_size
        self.step_size = 1
        self.testing = testing
        self.n_classes = n_classes

    def forward(self, x):
        if not self.testing:
            return self.model(x)
        else:
            # Pad the input image to ensure patches at the border are handled
            padding = (self.patch_size - self.step_size) // 2
            x_padded = nn.functional.pad(x, (padding+1, padding, padding+1, padding))

            # Unfold the input tensor to create patches
            patches = x_padded.unfold(2, self.patch_size, self.step_size).unfold(3, self.patch_size, self.step_size)
            
            s = 2
            b, _, h, w = x.shape
            y = torch.zeros(b, self.n_classes, h, w, device=x.device, dtype=x.dtype)
            for b in tqdm(range(0, x.size(0))):
                for i in range(0, x.size(3)//s):
                    # Reshape patches to (batch_size * num_patches_y * num_patches_x, channels, patch_size, patch_size)
                    p = patches[b, :, :, i*s:(i+1)*s].permute(1, 2, 0, 3, 4).contiguous()
                    p = p.view(-1, x.size(1), self.patch_size, self.patch_size)

                    yp = self.model(p)

                    # Reshape the output back to the patch grid
                    y[b, ... , i*s:(i+1)*s] = yp.view(x.size(2), s, -1).permute(2, 0, 1)
                    del yp, p
                
            return y
