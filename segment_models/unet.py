import torch.nn as nn
import torch

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock, self).__init__()

        # self.up = nn.Upsample(scale_factor=2)
        self.up = nn.ConvTranspose2d(in_channels//2,in_channels//2,(2,2),2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        out = self.up(x)
        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)

class UpBlockPad(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, bilinear=True, activation='ReLU'):
        super(UpBlockPad, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.nConvs(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, shallow=0):
        '''
        n_channels : number of channels of the input.
                        By default 3, because we have RGB images
        n_labels : number of channels of the ouput.
                      By default 3 (2 labels + 1 for the background)
        '''
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shallow = shallow
        in_channels = 64
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.down1 = DownBlock(in_channels, in_channels*2, nb_Conv=2)
        self.up1 = UpBlockPad(in_channels*2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1))
        if self.shallow <= 2:
            self.down2 = DownBlock(in_channels*2, in_channels*4 if self.shallow < 2 else in_channels*2, nb_Conv=2)
            self.up2 = UpBlockPad(in_channels*4, in_channels, nb_Conv=2)
        if self.shallow <= 1:
            self.up3 = UpBlockPad(in_channels*8, in_channels*2, nb_Conv=2)
            self.down3 = DownBlock(in_channels*4, in_channels*8 if self.shallow == 0 else in_channels*4, nb_Conv=2)
        if self.shallow == 0:
            self.down4 = DownBlock(in_channels*8, in_channels*8, nb_Conv=2)
            self.up4 = UpBlockPad(in_channels*16, in_channels*4, nb_Conv=2)

        if n_classes == 1:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = None #nn.Softmax(dim=1)

    def forward(self, x):
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        if self.shallow <= 2:
            x3 = self.down2(x2)
            x = x3
        if self.shallow <= 1:
            x4 = self.down3(x3)
            x = x4
        if self.shallow == 0:
            x5 = self.down4(x4)
            x = self.up4(x5, x4)
        if self.shallow <= 1:
            x = self.up3(x, x3)
        if self.shallow <= 2:
            x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.outc(x)

        if self.last_activation is not None:
            x = self.last_activation(x)

        return x
