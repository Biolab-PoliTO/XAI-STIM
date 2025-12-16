import torch

"""
cbam
https://github.com/Jongchan/attention-module.git
"""

class BasicConv(torch.nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = torch.nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = torch.nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(torch.nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = torch.nn.Sequential(
            Flatten(),
            torch.nn.Linear(gate_channels, gate_channels // reduction_ratio),
            torch.nn.ReLU(),
            torch.nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = torch.nn.functional.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = torch.nn.functional.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = torch.nn.functional.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.nn.functional.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(torch.nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(torch.nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.nn.functional.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(torch.nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

"""
Xiao, 2022
"""
class CNN_CBAM(torch.nn.Module):
    def __init__(self, in_channels = 6, num_classes=2):
        """
        ...
        """
        super(CNN_CBAM, self).__init__()
        
        # Layer 1: Conv + BN + ReLU
        self.conv1 = torch.nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.cbam1 = CBAM(gate_channels=16, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)
        
        # Layer 2: Conv + BN + ReLU
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.cbam2 = CBAM(gate_channels=32, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)
        
        # Layer 3: Conv + BN + ReLU
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.cbam3 = CBAM(gate_channels=64, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)
        
        # Layer 4: Conv + BN + ReLU
        self.conv4 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(128)
        self.cbam4 = CBAM(gate_channels=128, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)
        
        # MaxPool 1
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 5: Conv + BN + ReLU
        self.conv5 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn5 = torch.nn.BatchNorm2d(128)
        self.cbam5 = CBAM(gate_channels=128, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)
        
        # MaxPool 2
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 6: Conv + BN + ReLU
        self.conv6 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = torch.nn.BatchNorm2d(128)
        self.cbam6 = CBAM(gate_channels=128, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)
        
        # MaxPool 3
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 7: Conv + BN + ReLU
        self.conv7 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn7 = torch.nn.BatchNorm2d(128)
        self.cbam7 = CBAM(gate_channels=128, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False)
        
        # Calculate the size after all conv and pooling layers
        # Input: 100x100 -> after 3 maxpool(2x2): 100//8 = 12 (rounded down from 12.5)
        # So final feature map size is approximately 12x12x128
        self.fc_input_size = 12 * 12 * 128
        
        # Fully Connected Layer
        self.fc = torch.nn.Linear(self.fc_input_size, num_classes)
        
    def forward(self, x):
        """
        ...
        """

        # Layer 1: Conv + BN + ReLU + CBAM
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.cbam1(x)
        
        # Layer 2: Conv + BN + ReLU + CBAM
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        x = self.cbam2(x)
        
        # Layer 3: Conv + BN + ReLU + CBAM
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.nn.functional.relu(x)
        x = self.cbam3(x)
        
        # Layer 4: Conv + BN + ReLU + CBAM
        x = self.conv4(x)
        x = self.bn4(x)
        x = torch.nn.functional.relu(x)
        x = self.cbam4(x)
        
        # MaxPool 1
        x = self.maxpool1(x)  # 100x100 -> 50x50
        
        # Layer 5: Conv + BN + ReLU + CBAM
        x = self.conv5(x)
        x = self.bn5(x)
        x = torch.nn.functional.relu(x)
        x = self.cbam5(x)
        
        # MaxPool 2
        x = self.maxpool2(x)  # 50x50 -> 25x25
        
        # Layer 6: Conv + BN + ReLU + CBAM
        x = self.conv6(x)
        x = self.bn6(x)
        x = torch.nn.functional.relu(x)
        x = self.cbam6(x)
        
        # MaxPool 3
        x = self.maxpool3(x)  # 25x25 -> 12x12 (rounded down from 12.5)
        
        # Layer 7: Conv + BN + ReLU + CBAM
        x = self.conv7(x)
        x = self.bn7(x)
        x = torch.nn.functional.relu(x)
        x = self.cbam7(x)
        
        # Flatten for fully connected layer
        x = x.view(x.size(0), -1)
        
        # Fully Connected Layer
        x = self.fc(x)
        
        return x, 0
    
def main():
    """
    ...
    """
    
    # Add parent directory to path to import from utils

    from torchinfo import summary
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model instance
    model = CNN_CBAM(in_channels = 6, num_classes=2)
    model.to(device)

    model_summary = summary(
        model, 
        input_size=(1, 6, 100, 100),
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