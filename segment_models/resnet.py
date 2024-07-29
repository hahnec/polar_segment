from torch import nn


class PatchResNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, patch_size=50):
        super(PatchResNet, self).__init__()

        from monai.networks.nets import resnet18
        self.model = resnet18(pretrained=False, spatial_dims=2, n_input_channels=n_channels, num_classes=n_classes)
        self.patch_size = patch_size
        self.step_size = 1

    def forward(self, x):

        # Pad the input image to ensure patches at the border are handled
        padding = (self.patch_size - self.step_size) // 2
        x_padded = nn.functional.pad(x, (padding+1, padding, padding+1, padding))

        # Unfold the input tensor to create patches
        patches = x_padded.unfold(2, self.patch_size, self.step_size).unfold(3, self.patch_size, self.step_size)

        # Calculate the number of patches in each dimension
        num_patches_y = patches.size(2)
        num_patches_x = patches.size(3)

        # Reshape patches to (batch_size * num_patches_y * num_patches_x, channels, patch_size, patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(-1, x.size(1), self.patch_size, self.patch_size)
        
        y = self.model(patches)

        # Reshape the output back to the patch grid
        y = y.view(x.size(0), num_patches_y, num_patches_x, -1).permute(0, 3, 1, 2)
        
        return y
