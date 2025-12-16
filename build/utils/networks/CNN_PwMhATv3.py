# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# # -----------------------------------------------------
# # PATCH EXTRACTOR
# # -----------------------------------------------------

# class PatchExtractor1D(nn.Module):
#     """Split 1D signal into non-overlapping patches."""
#     def __init__(self, patch_size):
#         super().__init__()
#         self.patch_size = patch_size

#     def forward(self, x):
#         """
#         x: (B, 1, L)
#         return: (B*Np, 1, patch_size), B, Np
#         """
#         B, C, L = x.shape
#         Np = L // self.patch_size

#         x = x[:, :, :Np * self.patch_size]     # trim if needed
#         x = x.view(B, C, Np, self.patch_size)  # (B, C, Np, P)
#         x = x.permute(0, 2, 1, 3)              # (B, Np, C, P)
#         x = x.reshape(B * Np, C, self.patch_size)

#         return x, B, Np


# # -----------------------------------------------------
# # PATCH CNN ENCODER (per patch)
# # -----------------------------------------------------

# class PatchCNNEncoder(nn.Module):
#     """
#     Extract a feature embedding from each patch using CNN.
#     Equivalent to ViT patch projection, but more expressive.
#     """
#     def __init__(self, in_channels, embed_dim, dropout=0.0):
#         super().__init__()

#         self.encoder = nn.Sequential(
#             nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
#             nn.GELU(),
#             nn.Conv1d(32, 32, kernel_size=3, padding=1),
#             nn.GELU(),
#             nn.AdaptiveAvgPool1d(4),
#             nn.Flatten(),  # (B, 32*4)
#             nn.Linear(32*4, embed_dim),
#             nn.GELU(),
#             nn.Dropout(dropout)
#         )

#     def forward(self, x):
#         return self.encoder(x)  # (B, embed_dim)


# # -----------------------------------------------------
# # TRANSFORMER ENCODER BLOCK
# # -----------------------------------------------------

# class TransformerBlock1D(nn.Module):
#     def __init__(self, dim, num_heads, mlp_ratio=1, dropout=0.1):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
#         self.drop1 = nn.Dropout(dropout)

#         self.norm2 = nn.LayerNorm(dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, int(dim * mlp_ratio)),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(int(dim * mlp_ratio), dim),
#             nn.Dropout(dropout),
#         )

#     def forward(self, x):
#         attn_out, attn_weights = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
#         x = x + self.drop1(attn_out)
#         x = x + self.mlp(self.norm2(x))
#         return x, attn_weights


# # -----------------------------------------------------
# # Full Hybrid Model
# # -----------------------------------------------------

# class CNN_PwMhATv3(nn.Module):
#     def __init__(
#         self,
#         input_length=10000,
#         patch_size=100,
#         embed_dim=128,
#         depth=1,
#         num_heads=4,
#         num_classes=2,
#         dropout=0.2,
#         use_cls_token=True,
#     ):
#         super().__init__()

#         self.patch_size = patch_size
#         self.num_patches = input_length // patch_size
#         self.use_cls_token = use_cls_token

#         # --- Patch splitting ---
#         self.patch_extractor = PatchExtractor1D(patch_size)

#         # --- Patch CNN encoder ---
#         self.patch_cnn = PatchCNNEncoder(
#             in_channels=1,
#             embed_dim=embed_dim,
#             dropout=dropout
#         )

#         # --- Positional embeddings ---
#         self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + (1 if use_cls_token else 0), embed_dim))

#         # --- CLS token ---
#         if use_cls_token:
#             self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

#         # --- Transformer encoder ---
#         self.blocks = nn.ModuleList([
#             TransformerBlock1D(embed_dim, num_heads, mlp_ratio=4, dropout=dropout)
#             for _ in range(depth)
#         ])

#         self.norm = nn.LayerNorm(embed_dim)

#         # --- Classification head ---
#         self.head = nn.Linear(embed_dim, num_classes)

#         nn.init.trunc_normal_(self.pos_embed, std=0.02)
#         if use_cls_token:
#             nn.init.trunc_normal_(self.cls_token, std=0.02)

#     def forward(self, x):
#         if len(x.shape) < 3:
#             x = x.unsqueeze(1)  # (B, 1, L)
#         # -------------------------------
#         # Patch splitting
#         # -------------------------------
#         patches, B, Np = self.patch_extractor(x)  
#         # patches: (B*Np, 1, patch_size)

#         # -------------------------------
#         # CNN on each patch
#         # -------------------------------
#         tokens = self.patch_cnn(patches)         # (B*Np, embed_dim)
#         tokens = tokens.view(B, Np, -1)          # (B, Np, embed_dim)

#         # -------------------------------
#         # Add CLS token
#         # -------------------------------
#         if self.use_cls_token:
#             cls = self.cls_token.expand(B, -1, -1)
#             tokens = torch.cat([cls, tokens], dim=1)

#         # -------------------------------
#         # Add positional encoding
#         # -------------------------------
#         tokens = tokens + self.pos_embed

#         # -------------------------------
#         # Transformer
#         # -------------------------------
#         attn_maps = []
#         for blk in self.blocks:
#             tokens, att = blk(tokens)
#             attn_maps.append(att)

#         tokens = self.norm(tokens)

#         # -------------------------------
#         # Pooling
#         # -------------------------------
#         if self.use_cls_token:
#             out = tokens[:, 0]           # CLS
#         else:
#             out = tokens.mean(dim=1)     # mean-pool

#         # -------------------------------
#         # Classification
#         # -------------------------------
#         logits = self.head(out)
#         return logits, attn_maps



# def main():
#     """
#     ...
#     """
   
#     from torchinfo import summary
    
#     # Set device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
    
#     # Create model instance
#     model = CNN_PwMhATv3()
#     model.to(device)

#     model_summary = summary(
#         model, 
#         input_size=(1, 10000),
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         verbose=1
#     )
    
#     # Count total parameters
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
#     print(f"\n2. PARAMETER COUNT:")
#     print("-" * 40)
#     print(f"Total parameters: {total_params:,}")
#     print(f"Trainable parameters: {trainable_params:,}")
#     print(f"Non-trainable parameters: {total_params - trainable_params:,}")

# if __name__ == "__main__":
#     main()


# """
# ProvenHybrid1D: reliable 1D CNN + Transformer hybrid for classification.

# - Patch-wise CNN encoder: each patch is encoded by a small CNN -> one token per patch.
# - Transformer encoder: pre-LN MultiheadAttention + MLP, with optional stochastic depth.
# - Use CLS token or mean pooling.
# """

# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Optional, Tuple, List


# # -------------------------
# # Utilities
# # -------------------------
# def trunc_normal_(tensor, mean=0., std=1e-2):
#     # small helper — uses normal_ then clips (simple)
#     with torch.no_grad():
#         return tensor.normal_(mean, std)


# class DropPath(nn.Module):
#     """Stochastic depth / DropPath per sample"""
#     def __init__(self, drop_prob: float = 0.0):
#         super().__init__()
#         self.drop_prob = drop_prob

#     def forward(self, x):
#         if self.drop_prob == 0. or not self.training:
#             return x
#         keep_prob = 1 - self.drop_prob
#         # shape: (batch, 1, 1) to broadcast across sequence and channels
#         shape = (x.shape[0],) + (1,) * (x.dim() - 1)
#         random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
#         random_tensor.floor_()  # binarize
#         return x.div(keep_prob) * random_tensor


# # -------------------------
# # Patch extractor (unfold-based safe)
# # -------------------------
# class PatchExtractor1D(nn.Module):
#     """
#     Split a 1D signal into non-overlapping patches using unfold.
#     Input: (B, C, L)
#     Output: patches (B * num_patches, C, patch_size), B, num_patches
#     """
#     def __init__(self, patch_size: int):
#         super().__init__()
#         self.patch_size = patch_size

#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
#         B, C, L = x.shape
#         # trim right if not divisible
#         n_patches = L // self.patch_size
#         if n_patches == 0:
#             raise ValueError("patch_size too large for input length")
#         x = x[:, :, : n_patches * self.patch_size]
#         # unfold-like reshape:
#         x = x.view(B, C, n_patches, self.patch_size)  # (B, C, Np, P)
#         x = x.permute(0, 2, 1, 3).contiguous()        # (B, Np, C, P)
#         patches = x.view(B * n_patches, C, self.patch_size)
#         return patches, B, n_patches


# # -------------------------
# # Patch CNN encoder (per patch)
# # -------------------------
# class PatchCNNEncoder(nn.Module):
#     """
#     A small per-patch CNN producing one token per patch.
#     Returns (B*num_patches, embed_dim)
#     """
#     def __init__(self, in_ch: int = 1, embed_dim: int = 64, hidden: int = 32, dropout: float = 0.1):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv1d(in_ch, hidden, kernel_size=5, padding=2, bias=True),
#             nn.GELU(),
#             nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, bias=True),
#             nn.GELU(),
#             nn.AdaptiveAvgPool1d(1),   # keep small temporal structure collapsed to vector
#             nn.Flatten(1),
#             nn.Linear(hidden, embed_dim),
#             nn.GELU(),
#             nn.Dropout(dropout)
#         )
#         self._init_weights()

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.net(x)  # (B*num_patches, embed_dim)


# # -------------------------
# # Transformer components
# # -------------------------
# class TransformerBlock(nn.Module):
#     def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
#                  dropout: float = 0.1, attn_dropout: float = 0.1, drop_path: float = 0.0,
#                  layerscale: bool = False, ls_init: float = 1e-4):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
#                                           dropout=attn_dropout, batch_first=True)
#         self.attn_drop = nn.Dropout(dropout)

#         self.norm2 = nn.LayerNorm(dim)
#         mlp_hidden = int(dim * mlp_ratio)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, mlp_hidden),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(mlp_hidden, dim),
#             nn.Dropout(dropout)
#         )

#         self.drop_path = DropPath(drop_path)
#         self.use_layerscale = layerscale
#         if layerscale:
#             self.gamma1 = nn.Parameter(ls_init * torch.ones(dim))
#             self.gamma2 = nn.Parameter(ls_init * torch.ones(dim))
#         else:
#             self.gamma1 = None
#             self.gamma2 = None

#     def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
#         # Pre-LN attention
#         x_norm = self.norm1(x)
#         attn_out, attn_w = self.attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask, need_weights=True)
#         attn_out = self.attn_drop(attn_out)
#         if self.use_layerscale:
#             x = x + self.drop_path(self.gamma1.unsqueeze(0).unsqueeze(0) * attn_out)
#         else:
#             x = x + self.drop_path(attn_out)

#         # MLP
#         y = self.mlp(self.norm2(x))
#         if self.use_layerscale:
#             x = x + self.drop_path(self.gamma2.unsqueeze(0).unsqueeze(0) * y)
#         else:
#             x = x + self.drop_path(y)

#         return x, attn_w


# # -------------------------
# # Final hybrid model
# # -------------------------
# class CNN_PwMhATv3(nn.Module):
#     def __init__(
#         self,
#         input_length: int = 10000,
#         patch_size: int = 100,
#         in_channels: int = 1,
#         embed_dim: int = 128,
#         patch_hidden: int = 64,
#         depth: int = 4,
#         num_heads: int = 4,
#         mlp_ratio: float = 4.0,
#         dropout: float = 0.15,
#         attn_dropout: float = 0.15,
#         drop_path_rate: float = 0.2,
#         use_cls_token: bool = True,
#         num_classes: int = 2,
#         layerscale: bool = False,
#     ):
#         """
#         input_length=input_length,
#         patch_size=patch_size,
#         in_channels=1,
#         embed_dim=128,
#         patch_hidden=64,
#         depth=4,
#         num_heads=4,
#         mlp_ratio=4.0,
#         dropout=0.15,
#         attn_dropout=0.15,
#         drop_path_rate=0.2,
#         use_cls_token=True,
#         num_classes=num_classes,
#         layerscale=True
#         """
#         super().__init__()

#         # sanity
#         assert input_length % patch_size == 0, "input_length must be divisible by patch_size for this simple impl"
#         self.patch_size = patch_size
#         self.num_patches = input_length // patch_size
#         self.use_cls_token = use_cls_token
#         self.embed_dim = embed_dim

#         # Patch extractor + per-patch CNN
#         self.patch_extractor = PatchExtractor1D(patch_size)
#         self.patch_cnn = PatchCNNEncoder(in_ch=in_channels, embed_dim=embed_dim,
#                                          hidden=patch_hidden, dropout=dropout)

#         # cls token + positional embedding
#         if use_cls_token:
#             self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         else:
#             self.register_parameter('cls_token', None)

#         self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + (1 if use_cls_token else 0), embed_dim))
#         trunc_normal_(self.pos_embed, std=0.02)
#         if use_cls_token:
#             trunc_normal_(self.cls_token, std=1e-6)

#         # Transformer blocks
#         # linear schedule for drop_path
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
#         blocks: List[nn.Module] = []
#         for i in range(depth):
#             blocks.append(
#                 TransformerBlock(
#                     dim=embed_dim,
#                     num_heads=num_heads,
#                     mlp_ratio=mlp_ratio,
#                     dropout=dropout,
#                     attn_dropout=attn_dropout,
#                     drop_path=dpr[i],
#                     layerscale=layerscale
#                 )
#             )
#         self.blocks = nn.ModuleList(blocks)
#         self.norm = nn.LayerNorm(embed_dim)

#         # head
#         self.head = nn.Linear(embed_dim, num_classes)
#         nn.init.xavier_uniform_(self.head.weight)
#         if self.head.bias is not None:
#             nn.init.zeros_(self.head.bias)

#         # small init for projection layers inside patch cnn already done

#     def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
#         """
#         x: (B, in_channels, L)
#         returns: logits (B, num_classes), aux {attn_maps, tokens}
#         """
#         if len(x.shape) < 3:
#             x = x.unsqueeze(1)  # (B, 1, L)
#         patches, B, Np = self.patch_extractor(x)           # (B*Np, C, P)
#         tokens = self.patch_cnn(patches)                   # (B*Np, embed_dim)
#         tokens = tokens.view(B, Np, -1)                    # (B, Np, embed_dim)

#         if self.use_cls_token:
#             cls = self.cls_token.expand(B, -1, -1)         # (B,1,embed_dim)
#             tokens = torch.cat([cls, tokens], dim=1)       # (B, 1+Np, embed_dim)

#         tokens = tokens + self.pos_embed.to(tokens.device)

#         attn_maps = []
#         for blk in self.blocks:
#             tokens, attn_w = blk(tokens, key_padding_mask=key_padding_mask)
#             attn_maps.append(attn_w)

#         tokens = self.norm(tokens)

#         if self.use_cls_token:
#             pooled = tokens[:, 0]
#         else:
#             pooled = tokens.mean(dim=1)

#         logits = self.head(pooled)
#         aux = dict(tokens=tokens, attn_maps=attn_maps, pooled=pooled)
#         return logits, aux


# # -------------------------
# # Small factory
# # -------------------------
# def proven_hybrid_1d_tiny(input_length=10000, patch_size=200, num_classes=2):
#     return CNN_PwMhATv3(
#         input_length=input_length,
#         patch_size=patch_size,
#         in_channels=1,
#         embed_dim=64,
#         patch_hidden=32,
#         depth=2,
#         num_heads=2,
#         mlp_ratio=2.5,
#         dropout=0.1,
#         attn_dropout=0.1,
#         drop_path_rate=0.05,
#         use_cls_token=True,
#         num_classes=num_classes,
#         layerscale=False
#     )


# def proven_hybrid_1d_small(input_length=10000, patch_size=200, num_classes=2):
#     return CNN_PwMhATv3(
#         input_length=input_length,
#         patch_size=patch_size,
#         in_channels=1,
#         embed_dim=96,
#         patch_hidden=48,
#         depth=3,
#         num_heads=3,
#         mlp_ratio=3.0,
#         dropout=0.12,
#         attn_dropout=0.12,
#         drop_path_rate=0.1,
#         use_cls_token=True,
#         num_classes=num_classes,
#         layerscale=True
#     )


# def proven_hybrid_1d_base(input_length=10000, patch_size=200, num_classes=2):
#     return CNN_PwMhATv3(
#         input_length=input_length,
#         patch_size=patch_size,
#         in_channels=1,
#         embed_dim=128,
#         patch_hidden=64,
#         depth=4,
#         num_heads=4,
#         mlp_ratio=4.0,
#         dropout=0.15,
#         attn_dropout=0.15,
#         drop_path_rate=0.2,
#         use_cls_token=True,
#         num_classes=num_classes,
#         layerscale=True
#     )

# def main():
#     """
#     ...
#     """
   
#     from torchinfo import summary
    
#     # Set device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
    
#     # Create model instance
#     model = CNN_PwMhATv3()
#     model.to(device)

#     model_summary = summary(
#         model, 
#         input_size=(1, 10000),
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         verbose=1
#     )
    
#     # Count total parameters
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
#     print(f"\n2. PARAMETER COUNT:")
#     print("-" * 40)
#     print(f"Total parameters: {total_params:,}")
#     print(f"Trainable parameters: {trainable_params:,}")
#     print(f"Non-trainable parameters: {total_params - trainable_params:,}")

# if __name__ == "__main__":
#     main()


import torch
import torch.nn as nn

# -----------------------------------------------------
# PATCH EXTRACTOR
# -----------------------------------------------------

class PatchExtractor1D(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        B, C, L = x.shape
        n_patches = L // self.patch_size
        x = x[:, :, :n_patches * self.patch_size]
        x = x.view(B, C, n_patches, self.patch_size)
        x = x.permute(0, 2, 1, 3).reshape(B * n_patches, C, self.patch_size)
        return x, B, n_patches
    
def patch_dropout(x, drop_prob=0.1, training=True):
    """
    x: (B, N, D)  # B=batch, N=number of patches, D=embedding dim
    drop_prob: probability to drop each patch
    training: only drop patches during training
    """
    if not training or drop_prob == 0.0:
        return x

    B, N, D = x.shape

    # Create a dropout mask for tokens
    mask = torch.rand(B, N, device=x.device) > drop_prob  # True = keep, False = drop
    mask = mask.unsqueeze(-1)  # (B, N, 1)
    x = x * mask  # zero-out dropped tokens

    # Optional: scale to keep expectation same (like standard dropout)
    x = x / (1.0 - drop_prob)

    return x


# -----------------------------------------------------
# CNN PATCH ENCODER
# -----------------------------------------------------

class PatchCNN(nn.Module):
    def __init__(self, in_ch=1, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(64 * 4, embed_dim)
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------------------------------
# MULTIHEAD ATTENTION BLOCK
# -----------------------------------------------------

class MHABlock(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim,
            heads,
            dropout=dropout,
            batch_first=True
        )
        self.drop = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        # x: (B, N, D)
        x_norm = self.norm(x)
        x_att, attn = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop(x_att)
        x = x + self.mlp(x)
        return x, attn


# -----------------------------------------------------
# ATTENTION POOLING
# -----------------------------------------------------

class AttnPool(nn.Module):
    """
    Learns to weight tokens before pooling.
    Excellent for removing artifact patches.
    """
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x):
        w = torch.softmax(self.score(x), dim=1)
        return (w * x).sum(dim=1)


# -----------------------------------------------------
# HYBRID CNN + MHA MODEL
# -----------------------------------------------------

class CNN_PwMhATv3(nn.Module):

    def __init__(
        self,
        input_length=10000,
        patch_size=200,
        embed_dim=128,
        num_heads=2,
        depth=1,
        num_classes=2,
        dropout=0.1
    ):
        super().__init__()

        self.extractor = PatchExtractor1D(patch_size)

        self.encoder = PatchCNN(
            in_ch=1,
            embed_dim=embed_dim
        )

        n_patches = input_length // patch_size
        self.pos_embed = nn.Parameter(
            torch.randn(1, n_patches, embed_dim) * 0.02
        )

        self.blocks = nn.ModuleList([
            MHABlock(
                dim=embed_dim,
                heads=num_heads,
                dropout=dropout
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self.pool = AttnPool(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):

        if x.dim() == 2:
            x = x.unsqueeze(1)

        patches, B, N = self.extractor(x)
        tokens = self.encoder(patches)

        tokens = tokens.view(B, N, -1)
        tokens = tokens + self.pos_embed

        # -------------------------------
        # Apply patch dropout
        # -------------------------------
        tokens = patch_dropout(tokens, drop_prob=0.1, training=self.training)

        attn_maps = []
        for blk in self.blocks:
            tokens, att = blk(tokens)
            attn_maps.append(att)

        tokens = self.norm(tokens)

        pooled = self.pool(tokens)

        logits = self.classifier(pooled)

        return logits, attn_maps

def main():
    """
    ...
    """
   
    from torchinfo import summary
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model instance
    model = CNN_PwMhATv3(input_length=10000)
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
