import torch
import torch.nn as nn
import torch.nn.functional as F

"""
ConvMixer1D
Adapted from ConvMixer (originally for images) to work with 1D input.
Input shape: (batch_size, in_channels, length) e.g., (1, 1, 10000)

https://arxiv.org/abs/2201.09792
"""

class ConvMixerBlock1D(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.depthwise = nn.Conv1d(dim, dim, kernel_size, groups=dim, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(dim)
        self.gelu1 = nn.GELU()
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(dim)
        self.gelu2 = nn.GELU()

    def forward(self, x):
        out = self.depthwise(x)
        out = self.gelu1(out)
        out = self.bn1(out)
        x = x + out  # Residual connection
        out = self.pointwise(x)
        out = self.gelu2(out)
        out = self.bn2(out)
        return out

class ConvMixer1D(nn.Module):
    def __init__(self, in_channels=1, dim=128, depth=3, patch_size=200, kernel_size=5, num_classes=2, input_length=10000):
        super().__init__()
        self.patch_embed = nn.Conv1d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.bn = nn.BatchNorm1d(dim)
        self.gelu = nn.GELU()
        self.blocks = nn.Sequential(*[
            ConvMixerBlock1D(dim, kernel_size) for _ in range(depth)
        ])
        # Calculate output length after patch embedding
        self.n_patches = input_length // patch_size
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1, 10000)
        x = self.patch_embed(x)
        x = self.bn(x)
        x = self.gelu(x)
        x = self.blocks(x)
        x = self.head(x)
        return x, 0

def main():
    from torchinfo import summary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = ConvMixer1D(in_channels=1, dim=128, depth=4, patch_size=200, kernel_size=5, num_classes=2, input_length=10000)
    model.to(device)
    model_summary = summary(
        model,
        input_size=(1, 1, 10000),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        verbose=1
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n2. PARAMETER COUNT:")
    print("-" * 40)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")

    # import pandas as pd
    # from torchinfo import summary
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Using device: {device}")

    # results = []
    # for dim in [32, 64, 128, 256]:
    #     for patch_size in range(5, 201, 5):
    #         for depth in range(1, 11):
    #             model = ConvMixer1D(
    #                 in_channels=1,
    #                 dim=dim,
    #                 depth=depth,
    #                 patch_size=patch_size,
    #                 kernel_size=5,
    #                 num_classes=2,
    #                 input_length=10000
    #             )
    #             model.to(device)
    #             total_params = sum(p.numel() for p in model.parameters())
    #             results.append({
    #                 'dim': dim,
    #                 'patch_size': patch_size,
    #                 'depth': depth,
    #                 'total_params': total_params
    #             })

    # df = pd.DataFrame(results)
    # df.to_excel('convmixer1d_param_table.xlsx', index=False)
    # print("Saved parameter table to convmixer1d_param_table.xlsx")

if __name__ == "__main__":
    main()
