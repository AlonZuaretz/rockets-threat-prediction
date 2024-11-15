import torch.nn as nn

from transformer_block import TransformerBlock


class ArticlesNN(nn.Module):
    def __init__(self, embedding_dim=256, num_heads=4, hidden_dim=512, num_transformers=3):
        super(ArticlesNN, self).__init__()
        # Separate transformers for articles and threats
        self.transformers = nn.ModuleList(
            [TransformerBlock(embedding_dim, num_heads, hidden_dim) for _ in range(num_transformers)])

        # Pooling layer
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, embeddings):
        # Article transformers flow
        for transformer in self.transformers:
            embeddings = transformer(embeddings)

        output = self.pooling(embeddings)
        return output


class ThreatsNN(nn.Module):
    def __init__(self, embedding_dim=256, num_heads=4, hidden_dim=512, num_transformers=3):
        super(ThreatsNN, self).__init__()
        # Separate transformers for articles and threats
        self.transformers = nn.ModuleList(
            [TransformerBlock(embedding_dim, num_heads, hidden_dim) for _ in range(num_transformers)])

        # Pooling layer
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, embeddings):
        # Article transformers flow
        for transformer in self.transformers:
            embeddings = transformer(embeddings)

        output = self.pooling(embeddings)
        return output


