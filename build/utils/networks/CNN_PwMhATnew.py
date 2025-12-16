import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import numpy as np
import os

class PatchExtractor(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
    
    def forward(self, x):
        B = x.size(0)  # batch size
        num_patches = x.size(-1) // self.patch_size
        
        # Reshape input to extract patches
        x = x.view(B, 1, num_patches, self.patch_size)
        x = x.view(B * num_patches, 1, self.patch_size)
        
        return x, B, num_patches

class PatchEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.MaxPool1d(kernel_size=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.MaxPool1d(kernel_size=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(32, out_channels, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x_out = self.block3(x2)
        return x_out

# class MultiHeadSelfAttention(nn.Module):
#     def __init__(self, feature_size, embed_dim, num_heads=2):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads

#         self.q_proj = nn.Linear(feature_size, embed_dim)
#         self.k_proj = nn.Linear(feature_size, embed_dim)
#         self.v_proj = nn.Linear(feature_size, embed_dim)

#         # Separate Q, K, V projections for each head
#         self.q_proj_heads = nn.ModuleList([nn.Linear(feature_size, self.head_dim) for _ in range(num_heads)])
#         self.k_proj_heads = nn.ModuleList([nn.Linear(feature_size, self.head_dim) for _ in range(num_heads)])
#         self.v_proj_heads = nn.ModuleList([nn.Linear(feature_size, self.head_dim) for _ in range(num_heads)])

#         self.norm = nn.LayerNorm(embed_dim)
#         self.scale = nn.Parameter(torch.tensor(1.0 / self.head_dim ** 0.5))

#         # Positional encoding
#         self.positional_encoding = None

#     def _get_positional_encoding(self, seq_len, embed_dim, device):
#         # Sinusoidal positional encoding
#         position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, embed_dim, 2, device=device).float() * (-np.log(10000.0) / embed_dim))
#         pe = torch.zeros(seq_len, embed_dim, device=device)
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         return pe

#     def forward(self, x, mask=None):
#         # x: (batch, seq_len, feature_size)
#         B, N, _ = x.shape

#         attn_outputs = []
#         attn_weights_list = []
#         attn_maps_list = []

#         Q = self.q_proj(x)  # (B, N, embed_dim)
#         K = self.k_proj(x)
#         V = self.v_proj(x)
#         for i in range(self.num_heads):
#             Qi = self.q_proj_heads[i](Q) # (B, N, head_dim)
#             Ki = self.k_proj_heads[i](K)
#             Vi = self.v_proj_heads[i](V)

#             # Compute attention scores
#             scores = torch.matmul(Qi, Ki.transpose(-2, -1)) * self.scale  # (B, N, N)
#             if mask is not None:
#                 scores = scores.masked_fill(mask == 0, float('-inf'))

#             attn_weights = F.softmax(scores, dim=-1)  # (B, N, N)
#             attn_map = torch.matmul(attn_weights, Vi)  # (B, N, head_dim)

#             attn_outputs.append(attn_map)
#             attn_weights_list.append(attn_weights)
#             attn_maps_list.append(attn_map)

#         # Concatenate heads
#         attn_map_cat = torch.cat(attn_outputs, dim=-1)  # (B, N, embed_dim)
#         output = self.norm(x + attn_map_cat)

#         # Stack attention weights and maps for all heads
#         attn_weights_stacked = torch.stack(attn_weights_list, dim=1)  # (B, num_heads, N, N)
#         # attn_map_stacked = torch.stack(attn_maps_list, dim=1)  # (B, num_heads, N, head_dim)

#         return output, attn_weights_stacked, attn_map_cat

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feature_size, embed_dim, num_heads=2, dropout=0.1):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Combine Q,K,V projections for efficiency
        self.qkv_proj = nn.Linear(feature_size, embed_dim * 3)

        # Output projection after concatenating all heads
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout on attention weights + output
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # LayerNorm BEFORE attention (pre-norm)
        self.norm = nn.LayerNorm(feature_size)

        # Scaling factor
        self.scale = (self.head_dim) ** -0.5

        # Positional encoding
        self.positional_encoding = None

    def _get_positional_encoding(self, seq_len, embed_dim, device):
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, device=device).float() * 
                             (-np.log(10000.0) / embed_dim))
        pe = torch.zeros(seq_len, embed_dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x, mask=None):
        """
        x: (B, N, feature_size)
        """
        B, N, _ = x.shape

        # Pre-norm improves stability
        x_norm = self.norm(x)

        if self.positional_encoding is None or self.positional_encoding.size(0) != N:
            self.positional_encoding = self._get_positional_encoding(N, x_norm.size(-1), x.device)

        x_pe = x_norm + self.positional_encoding  # add positional info

        # QKV linear projection
        qkv = self.qkv_proj(x_pe)  # (B, N, 3*embed_dim)
        Q, K, V = qkv.chunk(3, dim=-1)

        # Reshape: (B, N, num_heads, head_dim) → (B, num_heads, N, head_dim)
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, heads, N, N)

        if mask is not None:
            scores = scores.masked_fill(mask[:, None, None, :] == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention
        attn_output = torch.matmul(attn_weights, V)  # (B, heads, N, head_dim)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).reshape(B, N, -1)

        # Output projection + dropout
        out_map = self.out_proj(attn_output)
        out_map = self.proj_dropout(out_map)

        # Residual connection (post-projection)
        out = out_map + x

        return out, attn_weights, out_map

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()

        self.fc_block1 = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(p=dropout)
        )
        self.fc_block2 = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(p=dropout)
        )
        self.fc_block3 = nn.Linear(16, output_dim)

    def forward(self, x):
        fc_in = x
        x = self.fc_block1(x)
        x = self.fc_block2(x)
        x = self.fc_block3(x)
        return fc_in, x

class CNN_PwMhAT(nn.Module):
    def __init__(self, input_size=10000, patch_size=200, 
                 in_channels=1, out_channels=32,
                 embed_dim=32, num_heads=2,
                 output_dim=2, 
                 dropout=0.1):
        
        super().__init__()

        self.num_patches = input_size // patch_size
        self.patch_size = patch_size

        # Patch extractor block
        self.patch_extractor = PatchExtractor(
            patch_size = patch_size
        )

        self.norm = nn.LayerNorm(embed_dim)

        # Patch encoder block
        self.patch_encoder = PatchEncoder(
            in_channels = in_channels,
            out_channels = out_channels,
            dropout = dropout,
        )

        # Self-Attention block
        self.attention = MultiHeadSelfAttention(
            feature_size = embed_dim,
            embed_dim = embed_dim,
            num_heads = num_heads,
            dropout=dropout
        )

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 2*embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2*embed_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        # Final MLP
        self.fc = MLP(
            input_dim = embed_dim * self.num_patches,
            output_dim = output_dim,
            dropout = dropout
        )

    def forward(self, x):
        # Extract patches using the dedicated patch extractor
        x, B, num_patches = self.patch_extractor(x)

        patch_features = self.patch_encoder(x)  # (B*num_patches, C, 1 or T)
        patch_features = patch_features.squeeze(-1)
        patch_features = patch_features.view(B, num_patches, -1)
        patch_features = self.norm(patch_features)

        attn_out, attn_weights, attn_map = self.attention(patch_features)

        # Feed-Forward Network with residual connection
        ffn_out = self.ffn(self.norm2(attn_out))
        attn_out = attn_out + ffn_out

        flat_attn_out = attn_out.view(attn_out.size(0), -1)

        fc_in, out = self.fc(flat_attn_out)

        return out, {
            'patch_features': patch_features,
            'attention_outputs': attn_out,
            'attention_weights': attn_weights,
            'attention_map': attn_map,
            'fc_input': fc_in
        }

def main():
    """
    ...
    """
   
    from torchinfo import summary
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model instance
    model = CNN_PwMhAT(input_size=10000)
    model.to(device)

    model_summary = summary(
        model, 
        input_size=(1, 10000),
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