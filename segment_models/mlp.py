import torch
from torch import nn
from tqdm import tqdm


class MLP(torch.nn.Module):
    def __init__(self, n_channels=1, n_classes=2, hidden_dim=128):
        super(MLP, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.hidden_layer1 = torch.nn.Linear(n_channels, hidden_dim)
        self.hidden_layer2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.act_fun = torch.nn.ReLU() if True else torch.nn.GELU()
        self.output_layer = torch.nn.Linear(hidden_dim, n_classes)
        self.dropout = torch.nn.Dropout(.5)

        self._init_weights()

    def _init_weights(self):
        torch.nn.init.xavier_uniform_(self.hidden_layer1.weight)
        torch.nn.init.xavier_uniform_(self.hidden_layer2.weight)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)
        torch.nn.init.normal_(self.hidden_layer1.bias, std=1e-6)
        torch.nn.init.normal_(self.hidden_layer2.bias, std=1e-6)
        torch.nn.init.normal_(self.output_layer.bias, std=1e-6)

    def forward(self, x):
        dims = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, self.n_channels) if len(dims) == 4 else x
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.act_fun(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x.view(dims[0], *dims[2:], self.n_classes).permute(0, 3, 1, 2) if len(dims) == 4 else x


class PatchMLP(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, patch_size=50, testing=False):
        super(PatchMLP, self).__init__()

        self.model = MLP(n_channels=n_channels*patch_size**2, n_classes=n_classes)
        self.patch_size = patch_size
        self.step_size = 1
        self.testing = testing
        self.n_classes = n_classes

    def forward(self, x):
        if not self.testing:
            # merge spatial patch dimension with features
            x = x.flatten(1, -1)[..., None, None]
            return self.model(x).squeeze()
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

                    p = p.flatten(1, -1)[..., None, None]
                    yp = self.model(p)

                    # Reshape the output back to the patch grid
                    y[b, ... , i*s:(i+1)*s] = yp.view(x.size(2), s, -1).permute(2, 0, 1)
                    del yp, p
                
            return y


if __name__ == '__main__':

    n_classes = 2
    n_channels = 10
    model = MLP(n_channels, n_classes=n_classes)

    batch_size = 32
    x = torch.randn(batch_size, n_channels)
    output = model(x)
    print(output.shape)

    batch_size = 32
    x = torch.randn(batch_size, n_channels, 64, 64)
    output = model(x)
    print(output.shape)

    a = x.permute(0, 2, 3, 1).reshape(-1, n_channels).view(batch_size, *(64, 64), n_channels).permute(0, 3, 1, 2)
    assert torch.all(a == x)

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    axs[0].imshow(a[0, 0])
    axs[1].imshow(x[0, 0])
    plt.show()