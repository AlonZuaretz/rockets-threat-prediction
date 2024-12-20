import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, hidden_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        # Multi-head self-attention with the input dimension
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout)

        # Feed-forward layer with a final linear transformation to output_dim
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Layer normalization for input dimensions
        self.layer_norm_1 = nn.LayerNorm(input_dim)

        # Layer normalization for output dimensions
        self.layer_norm_2 = nn.LayerNorm(output_dim)

        self.projection = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Apply multi-head self-attention
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm_1(x + self.dropout(attn_output))

        # Apply feed-forward network
        ff_output = self.feed_forward(x)
        x = self.projection(x) + self.dropout(ff_output)
        x = self.layer_norm_2(x)

        return x