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

        self.num_patches = input_length // patch_size

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
