import torch
import torch.nn as nn
from functools import partial
import numpy as np
import scipy.signal as signal

"""
https://github.com/konradci/mer_classifier.git
"""

def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), nn.BatchNorm2d(out_channels))

class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)  # add padding based on the kernel_size

conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.blocks = nn.Identity()
        self.activate = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3):
        super().__init__(in_channels, out_channels)
        self.expansion = expansion
        self.downsampling = downsampling
        self.conv = conv

        # shortcut converts from in_channels to expanded_channels
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1, stride=self.downsampling, bias=False),
            nn.BatchNorm2d(self.expanded_channels)) if self.should_apply_shortcut else None

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels

class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            nn.ReLU(inplace=True),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )

class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        downsampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion, out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x
    
def spectrographer(tbl: np.ndarray, sample_rate, nperseg):
    assert tbl.dtype == np.float64
    _, _, spectrogram = signal.spectrogram(tbl[0, :], sample_rate, nperseg=nperseg)
    ret = np.zeros((tbl.shape[0], spectrogram.shape[0], spectrogram.shape[1]), dtype=np.float32)
    for i in range(tbl.shape[0]):
        _, _, spectrogram = signal.spectrogram(tbl[i, :], sample_rate, nperseg=nperseg,
                                               window=('tukey', .25), noverlap=None, nfft=None, detrend='constant',
                                               return_onesided=True, scaling='density', axis=-1, mode='psd'
                                               )
        ret[i, :, :] = spectrogram
    return ret
    
# resnet /w attention /w dropout
class ResNet_AT(nn.Module):
    def __init__(self, dim_0f=129, dim_1t=22, bsize=1):
        super(ResNet_AT, self).__init__()
        self.dim_0f = dim_0f
        self.dim_1t = dim_1t

        if isinstance(bsize, int):
            bsize = [bsize for _ in range(4)]

        self.head = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.resn1 = ResNetLayer(in_channels=64, out_channels=128, n=bsize[0])
        self.resn2 = ResNetLayer(in_channels=128, out_channels=256, n=bsize[1])
        self.resn3 = ResNetLayer(in_channels=256, out_channels=512, n=bsize[2])
        self.resn4 = ResNetLayer(in_channels=512, out_channels=1024, n=bsize[3])

        mid = 4000
        self.linmask1a = nn.Linear(128 * 65, mid)
        self.linmask1b = nn.Linear(mid, 128)
        self.linmask2a = nn.Linear(256 * 33, mid)
        self.linmask2b = nn.Linear(mid, 256)
        self.linmask3a = nn.Linear(512 * 17, mid)
        self.linmask3b = nn.Linear(mid, 512)
        self.linmask4a = nn.Linear(1024 * 9, mid)
        self.linmask4b = nn.Linear(mid, 1024)
        self.maskact = nn.Sigmoid()

        self.avg = nn.AdaptiveAvgPool2d((6, 6))
        self.decoder1 = nn.Linear(36 * 1024, 8192)
        self.decoder2 = nn.Linear(8192, 4096)
        # self.decoder3 = nn.Linear(4096, 1)
        self.decoder3 = nn.Linear(4096, 2)

        self.drop1 = nn.Dropout(p=0.25)
        self.drop2 = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor):
        self.head = self.head.float()
        x = torch.unsqueeze(x, 1)
        x = x.float()
        x = self.head(x)
        x = self.resn1(x)
        x = self.self_attention(x, self.linmask1a, self.linmask1b)
        x = self.resn2(x)
        x = self.self_attention(x, self.linmask2a, self.linmask2b)
        x = self.resn3(x)
        x = self.self_attention(x, self.linmask3a, self.linmask3b)
        x = self.resn4(x)
        x = self.self_attention(x, self.linmask4a, self.linmask4b)
        x = self.avg(x)
        x = x.view(x.shape[0], -1)
        x = self.decoder1(x)
        x = self.drop2(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        return x, 0

    # self attention in temporal domain
    def self_attention(self, x, att_layer1, att_layer2):
        _bs, _ch, _fd, _td = [x.shape[j] for j in range(4)]
        x = x.permute((0, 3, 1, 2))
        x = x.view(_bs, _td, _ch * _fd)
        am = att_layer1(x)
        am = self.drop1(am)
        am = att_layer2(am)
        am = self.maskact(am)
        am = torch.unsqueeze(am, 3)
        x = x.view(_bs, _td, _ch, _fd)
        am = am.expand(x.shape)
        x = torch.mul(x, am)
        x = x.permute((0, 2, 3, 1))
        return x
    
def main():
    """
    ...
    """
    from torchinfo import summary
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model instance
    input_size = (129, 51)
    model = ResNet_AT(dim_0f=input_size[0], dim_1t=input_size[1], bsize=1)
    model.to(device)

    model_summary = summary(
        model, 
        input_size=(1, input_size[0], input_size[1]),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        verbose=1
    )
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n2. PARAMETER COUNT:")
    print("-" * 40)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")

if __name__ == "__main__":
    main()
