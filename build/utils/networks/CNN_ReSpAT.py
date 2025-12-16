import torch
import math
import torch.nn as nn
import torch.nn.functional as F

"""
from Maged, 2024
As illustrated in Fig. 6, The input images are expected to have a
size of 227 x 227 pixels and three color channels. The model applies
two sets of CNN blocks, each with two convolutional layers followed by
batch normalization, ReLU activation, and max-pooling. Residual connections
is introduced after each CNN block, where the original input
is added to the output of the block. After the CNN blocks, the model
applies a spatial attention mechanism. The feature maps obtained after
the last CNN block are processed through a 1 x 1 convolutional layer
followed by batch normalization. Spatial attention is achieved using the
multi-head self-attention mechanism, which helps the model to focus
on relevant spatial regions. Multi-head self-attention divides the input
data into multiple heads, each representing a distinct subspace of the
feature space. For each head, the input is projected into three separate
Key, Query, and Value spaces. The attention scores are then computed using a 
scaled dot-product attention mechanism.

The scaled attention scores (A) are then used to compute the final attention 
output, O = A x V. The outputs from all the attention heads are concatenated
and linearly transformed to produce the final attention output. By
using multiple heads (2), the model can simultaneously attend to different
patterns and relationships, enabling it to effectively capture both local
and global dependencies within the input. The output feature maps
are then flattened and passed through two fully connected layers with
ReLU activation and a dropout layer in between. Additionally, there
are two separate residual connections at this stage. The output of the
second residual connections is added to the output of the second fully
connected layer. Finally, the output of the second fully connected layer
is batch-normalized and subjected to a soft thresholding function. The
purpose of applying soft thresholding at this stage is to further refine
the learned features before feeding them into the final classification
layer. By reducing noise or less relevant information, soft thresholding
can enhance the model's focus on more informative features, potentially
improving the classification performance and robustness of the
model.
SoftThresholding = sign(x) * max(|x| - 𝜏, 0)
where 𝜏 is a predefined threshold. Finally, the resulting tensor is passed
through the last fully connected layer to produce the final output.
"""


class CNN_ReSpAT(torch.nn.Module):
    def __init__(self, in_channels=3, num_classes=2, input_size=100):
        """
        ...
        """
        super(CNN_ReSpAT, self).__init__()
        # Basic configuration / hyperparameters
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.input_size = input_size

        # Convolutional blocks
        # Block 1: in_channels -> 16
        self.conv1_1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=0)
        self.conv1_2 = nn.Conv2d(16, 16, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(16)
        # ReLU
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 1x1 projection for residual if needed
        self.res1_conv = nn.Conv2d(in_channels, 16, kernel_size=3, padding=0)

        # Block 2: 16 -> 32
        self.conv2_1 = nn.Conv2d(16, 32, kernel_size=3, padding=0)
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(32)
        # ReLU
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # residual projection for block2
        self.res2_conv = nn.Conv2d(16, 32, kernel_size=3, padding=0)

        # 1x1 conv to prepare for attention (embed channels)
        self.embed_conv = nn.Conv2d(32, 32, kernel_size=1)
        self.embed_bn = nn.BatchNorm2d(32)

        # Multi-head self-attention over spatial positions
        # We'll treat each spatial location as a token: seq_len = H*W
        self.attn_embed_dim = 32
        self.num_heads = 2
        # Use batch_first=True for convenience: input shape (N, L, E)
        self.mha = nn.MultiheadAttention(embed_dim=self.attn_embed_dim, num_heads=self.num_heads, batch_first=True)

        # Fully connected layers after flattening attention output
        # compute flattened feature size after two poolings
        self.reduced_size = max(1, (input_size-6) // 4)  # two 2x pools -> /4
        self.flattened_dim = self.attn_embed_dim * (self.reduced_size ** 2)

        self.flat_bn = nn.BatchNorm1d(self.flattened_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(self.flattened_dim, 128)
        self.fc1_bn = nn.BatchNorm1d(128)

        # Residual projections for FC stage (two residual connections mentioned)
        self.res_fc1 = nn.Linear(self.flattened_dim, 128)

        # Soft-thresholding (tau). Learnable parameter
        self.tau = nn.Parameter(torch.tensor(0.1, dtype=torch.float32), requires_grad=True)

        # Final classification layer
        self.classifier = nn.Linear(128, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for Conv2d, Linear and BatchNorm layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # Kaiming/He init for linear layers
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x):
        """Forward pass.

        Returns:
            logits: Tensor of shape (N, num_classes)
            attn_map: attention weights tensor of shape (N, L, L) where L = H*W after embedding
        """
        # Block 1
        residual = x
        out = self.conv1_1(x)
        out = F.relu(self.bn1(self.conv1_2(out)))
        # out = self.pool1(out)
        residual = self.res1_conv(residual) # self.pool1(self.res1_conv(residual))
        out = self.pool1(out + residual)

        # Block 2
        residual = out
        out = self.conv2_1(out)
        out = F.relu(self.bn2(self.conv2_2(out)))
        # out = self.pool2(out)
        residual = self.res2_conv(residual) # self.pool2(self.res2_conv(residual))
        out = self.pool2(out + residual)

        # Embed for attention
        out = self.embed_bn(self.embed_conv(out))
        N, C, H, W = out.shape

        # Prepare for MHSA: (N, L, E)
        L = H * W
        # Shape to (N, L, E)
        out_flat = out.view(N, C, L).permute(0, 2, 1).contiguous()

        # Apply multi-head attention (queries=keys=values)
        attn_out, attn_weights = self.mha(out_flat, out_flat, out_flat)
        # attn_out: (N, L, E)
        # attn_weights: (N, L, L)

        # Reshape attention output back to (N, C, H, W)
        attn_out_reshaped = attn_out.permute(0, 2, 1).contiguous().view(N, C, H, W)

        # Flatten for FC layers
        flat = attn_out_reshaped.view(N, -1)

        # FC stage with two residual connections
        res1 = self.res_fc1(flat)
        flat = self.flat_bn(flat)
        flat = self.dropout(flat)
        x1 = self.fc1_bn(self.fc1(flat))
        x1 = x1 + res1

        # Soft-thresholding: sign(x) * max(|x|-tau, 0)
        # ensure tau is broadcastable
        tau = torch.abs(self.tau)
        x2 = torch.sign(x1) * F.relu(torch.abs(x1) - tau)

        logits = self.classifier(x2)

        return logits, attn_weights


def main():
    """
    ...
    """
    from torchinfo import summary
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = CNN_ReSpAT(in_channels=3, num_classes=2, input_size=100)
    model.to(device)
    
    model_summary = summary(
        model,
        input_size=(1, 3, 100, 100),
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
    
