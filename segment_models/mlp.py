import torch


class MLP(torch.nn.Module):
    def __init__(self, n_channels=1, n_classes=2, hidden_dim=64):
        super(MLP, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.hidden_layer = torch.nn.Linear(n_channels, hidden_dim)  # Single hidden layer with 64 units
        self.relu = torch.nn.ReLU()  # ReLU activation function
        self.output_layer = torch.nn.Linear(hidden_dim, n_classes)  # Output layer

    def forward(self, x):
        dims = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, self.n_channels) if len(dims) == 4 else x
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x.view(dims[0], *dims[2:], self.n_classes).permute(0, 3, 1, 2) if len(dims) == 4 else x


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