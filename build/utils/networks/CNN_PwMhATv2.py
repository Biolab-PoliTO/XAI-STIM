import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PatchExtractor(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        """
        x: (B, L)
        Returns:
            patches: (B * num_patches, 1, patch_size)
            B: original batch size
            num_patches: number of patches per sample
        """
        B, L = x.shape

        # Make sure the input is (B, 1, L)
        x = x.unsqueeze(1)

        # Use unfold to extract patches safely
        patches = x.unfold(dimension=2, size=self.patch_size, step=self.patch_size)
        # Result: (B, 1, num_patches, patch_size)

        num_patches = patches.size(2)

        # Reshape to (B * num_patches, 1, patch_size)
        patches = patches.permute(0, 2, 1, 3).reshape(B * num_patches, 1, self.patch_size)

        return patches, B, num_patches

# ---------- PatchEncoder with init + smaller dropout ----------
# class PatchEncoder(nn.Module):
#     def __init__(self, in_channels, out_channels, dropout=0.05):
#         super().__init__()
#         self.block1 = nn.Sequential(
#             nn.Conv1d(in_channels, 32, kernel_size=5, stride=1, padding=2),
#             nn.GELU(),
#             nn.Dropout(p=dropout),
#             nn.MaxPool1d(kernel_size=2)
#         )
#         self.block2 = nn.Sequential(
#             nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2),
#             nn.GELU(),
#             nn.Dropout(p=dropout),
#             nn.MaxPool1d(kernel_size=2)
#         )
#         self.block3 = nn.Sequential(
#             nn.Conv1d(32, out_channels, kernel_size=5, stride=1, padding=2),
#             nn.GELU(),
#             nn.Dropout(p=dropout),
#             nn.AdaptiveAvgPool1d(1)
#         )
#         self.reset_parameters()

#     def reset_parameters(self):
#         # Kaiming init for convs
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)

#     def forward(self, x):
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         return x

class PatchEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(64, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(4)   # <-- keep 4 features instead of collapsing to 1
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.flatten(1)    # B, out_channels * 4
        return x


# ---------- MLP using LayerNorm (safer) ----------
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.05):
        super().__init__()
        
        # hidden = min(128, max(32, input_dim // 4))
        # self.fc_block1 = nn.Linear(input_dim, hidden)
        # self.ln1 = nn.LayerNorm(hidden)
        # self.fc_block2 = nn.Linear(hidden, hidden // 2)
        # self.ln2 = nn.LayerNorm(hidden // 2)
        # self.out = nn.Linear(hidden // 2, output_dim)
        hidden = 32
        self.fc_block1 = nn.Linear(input_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc_block2 = nn.Linear(hidden, hidden // 2)
        self.ln2 = nn.LayerNorm(hidden // 2)
        self.out = nn.Linear(hidden // 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_block1.weight)
        nn.init.zeros_(self.fc_block1.bias)
        nn.init.xavier_uniform_(self.fc_block2.weight)
        nn.init.zeros_(self.fc_block2.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        # x: (B, feat)
        x0 = x
        x = self.fc_block1(x)
        x = self.ln1(self.act(x))
        x = self.dropout(x)

        x = self.fc_block2(x)
        x = self.ln2(self.act(x))
        x = self.dropout(x)

        out = self.out(x)
        return x0, out


# ---------- MultiHeadSelfAttention safe tweaks ----------
# class MultiHeadSelfAttention(nn.Module):
#     def __init__(self, feature_size, embed_dim, num_heads=2, dropout=0.05):
#         super().__init__()
#         assert embed_dim % num_heads == 0
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         self.qkv_proj = nn.Linear(feature_size, embed_dim * 3)
#         self.out_proj = nn.Linear(embed_dim, embed_dim)
#         self.attn_dropout = nn.Dropout(dropout)
#         self.proj_dropout = nn.Dropout(dropout)
#         self.norm = nn.LayerNorm(feature_size)
#         self.scale = (self.head_dim) ** -0.5
#         self.positional_encoding = None
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.qkv_proj.weight)
#         nn.init.zeros_(self.qkv_proj.bias)
#         nn.init.xavier_uniform_(self.out_proj.weight)
#         nn.init.zeros_(self.out_proj.bias)

#     def _get_positional_encoding(self, seq_len, embed_dim, device):
#         # sinusoidal
#         pe = torch.zeros(seq_len, embed_dim, device=device)
#         position = torch.arange(0, seq_len, device=device).unsqueeze(1).float()
#         div_term = torch.exp(torch.arange(0, embed_dim, 2, device=device).float() * (-math.log(10000.0) / embed_dim))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         return pe

#     def forward(self, x, mask=None):
#         B, N, C = x.shape
#         x_norm = self.norm(x)

#         # positional encoding: match feature dim (C)
#         if (self.positional_encoding is None) or (self.positional_encoding.size(0) != N) or (self.positional_encoding.device != x.device):
#             self.positional_encoding = self._get_positional_encoding(N, C, x.device)

#         x_pe = x_norm + self.positional_encoding

#         qkv = self.qkv_proj(x_pe)   # (B,N,3*embed)
#         Q, K, V = qkv.chunk(3, dim=-1)

#         Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,N,D)
#         K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
#         V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

#         scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B,H,N,N)
#         if mask is not None:
#             scores = scores.masked_fill(mask[:, None, None, :] == 0, float("-inf"))

#         attn = F.softmax(scores, dim=-1)
#         attn = self.attn_dropout(attn)

#         attn_out = torch.matmul(attn, V)  # (B,H,N,D)
#         attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, -1)  # (B,N,embed)

#         out_map = self.out_proj(attn_out)
#         out_map = self.proj_dropout(out_map)

#         out = out_map + x   # residual
#         return out, attn, out_map

class MHAWrapper(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # KEEP THIS
        )
        self.dropout = nn.Dropout(dropout)

        self.positional_encoding = None

    def _positional_encoding(self, seq_len, embed_dim, device):
        pe = torch.zeros(seq_len, embed_dim, device=device)
        position = torch.arange(seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, device=device) *
                             (-np.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x, mask=None):
        """
        x: (B, N, E)
        """
        B, N, E = x.shape

        # Pre-norm
        x_norm = self.norm(x)

        # Positional encoding
        if self.positional_encoding is None or self.positional_encoding.size(0) != N:
            self.positional_encoding = self._positional_encoding(N, E, x.device)

        x_pe = x_norm + self.positional_encoding

        # Built-in MHA (batch_first=True)
        attn_out, attn_weights = self.mha(
            x_pe, x_pe, x_pe,      # Q, K, V
            need_weights=True,
            attn_mask=None,        # optional
            key_padding_mask=None  # optional
        )

        attn_out = self.dropout(attn_out)

        # Residual connection
        out = x + attn_out

        return out, attn_weights, attn_out

class CNN_PwMhATv2(nn.Module):
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

        self.norm = nn.LayerNorm(embed_dim*4)

        # Patch encoder block
        self.patch_encoder = PatchEncoder(
            in_channels = in_channels,
            out_channels = out_channels,
            dropout = dropout,
        )

        # Self-Attention block
        # self.attention = MultiHeadSelfAttention(
        #     feature_size = embed_dim,
        #     embed_dim = embed_dim,
        #     num_heads = num_heads,
        #     dropout=dropout
        # )
        self.attention = MHAWrapper(
            embed_dim = 4*embed_dim,
            num_heads = num_heads,
            dropout = dropout
        )

        self.ffn = nn.Sequential(
            nn.Linear(4*embed_dim, 4*embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4*embed_dim, 4*embed_dim),
        )
        self.norm2 = nn.LayerNorm(4*embed_dim)

        # Final MLP
        self.fc = MLP(
            input_dim = 4*embed_dim, # * self.num_patches,
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

        # flat_attn_out = attn_out.view(attn_out.size(0), -1)

        flat_attn_out = attn_out.mean(dim=1)


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
    model = CNN_PwMhATv2(input_size=10000)
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