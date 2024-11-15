import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        # Multi-head self-attention and feed-forward layer
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Apply multi-head self-attention
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm(x + self.dropout(attn_output))
        # Apply feed-forward network
        ff_output = self.feed_forward(x)
        x = self.layer_norm(x + self.dropout(ff_output))
        return x