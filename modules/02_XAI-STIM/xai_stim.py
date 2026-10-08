"""
XAI-STIM neural-network architecture.

The network classifies one-second, 10-kHz MER windows by encoding 200-sample
patches, relating the patch embeddings with multi-head self-attention, and
feeding the flattened representation to a small MLP.

Author: Fabrizio Sciscenti
        H-MoVe, Department of Electronics and Telecommunications, Politecnico di Torino
        fabrizio.sciscenti@polito.it
Last updated: 7 October 2026
"""

import torch
from torch import nn
from torch.nn import functional as F


class PatchExtractor(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        batch_size = x.size(0)
        number_of_patches = x.size(-1) // self.patch_size
        x = x.reshape(batch_size * number_of_patches, 1, self.patch_size)
        return x, batch_size, number_of_patches


class PatchEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(32, out_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        return self.block3(self.block2(self.block1(x)))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feature_size, embed_dim, num_heads=2):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(feature_size, embed_dim)
        self.k_proj = nn.Linear(feature_size, embed_dim)
        self.v_proj = nn.Linear(feature_size, embed_dim)
        self.q_proj_heads = nn.ModuleList(
            nn.Linear(feature_size, self.head_dim) for _ in range(num_heads)
        )
        self.k_proj_heads = nn.ModuleList(
            nn.Linear(feature_size, self.head_dim) for _ in range(num_heads)
        )
        self.v_proj_heads = nn.ModuleList(
            nn.Linear(feature_size, self.head_dim) for _ in range(num_heads)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.scale = nn.Parameter(torch.tensor(1.0 / self.head_dim**0.5))

    def forward(self, x, mask=None):
        query, key, value = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        outputs, weights = [], []
        for index in range(self.num_heads):
            q = self.q_proj_heads[index](query)
            k = self.k_proj_heads[index](key)
            v = self.v_proj_heads[index](value)
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float("-inf"))
            attention = F.softmax(scores, dim=-1)
            weights.append(attention)
            outputs.append(torch.matmul(attention, v))

        attention_map = torch.cat(outputs, dim=-1)
        # Residual fusion used by miscellaneous/utils/nns/CNN_PwMhAT.py during
        # model development and training.
        output = self.norm(x + attention_map)
        return output, torch.stack(weights, dim=1), attention_map


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        self.fc_block1 = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(dropout)
        )
        self.fc_block2 = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(), nn.BatchNorm1d(16), nn.Dropout(dropout)
        )
        self.fc_block3 = nn.Linear(16, output_dim)

    def forward(self, x):
        fc_input = x
        x = self.fc_block3(self.fc_block2(self.fc_block1(x)))
        return fc_input, x


class XAI_STIM(nn.Module):
    """Checkpoint-compatible CNN_PwMhAT classifier used by XAI-STIM."""

    def __init__(
        self,
        input_size=10000,
        patch_size=200,
        in_channels=1,
        out_channels=32,
        embed_dim=32,
        num_heads=2,
        output_dim=2,
        dropout=0.1,
    ):
        super().__init__()
        if input_size % patch_size:
            raise ValueError("input_size must be divisible by patch_size")
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_patches = input_size // patch_size
        self.patch_extractor = PatchExtractor(patch_size)
        self.patch_encoder = PatchEncoder(in_channels, out_channels, dropout)
        self.attention = MultiHeadSelfAttention(embed_dim, embed_dim, num_heads)
        self.fc = MLP(embed_dim * self.num_patches, output_dim, dropout)

    def forward(self, x):
        if x.ndim != 2 or x.size(1) != self.input_size:
            raise ValueError(f"Expected input shape (batch, {self.input_size})")
        x, batch_size, number_of_patches = self.patch_extractor(x)
        patch_features = self.patch_encoder(x).squeeze(-1)
        patch_features = patch_features.reshape(batch_size, number_of_patches, -1)
        attention_output, attention_weights, attention_map = self.attention(patch_features)
        fc_input, output = self.fc(attention_output.reshape(batch_size, -1))
        return output, {
            "patch_features": patch_features,
            "attention_outputs": attention_output,
            "attention_weights": attention_weights,
            "attention_map": attention_map,
            "fc_input": fc_input,
        }
