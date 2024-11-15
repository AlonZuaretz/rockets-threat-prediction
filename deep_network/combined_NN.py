import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_block import TransformerBlock


class CombinedNN(nn.Module):
    def __init__(self, embedding_dim=256, num_heads=4, hidden_dim=512, num_transformers=3):
        super(CombinedNN, self).__init__()
        # Combined transformers for merged data after cross-correlation
        self.transformers = nn.ModuleList(
            [TransformerBlock(embedding_dim, num_heads, hidden_dim) for _ in range(num_transformers)])
        # Pooling layer
        self.pooling = nn.AdaptiveAvgPool1d(1)
        # Feed-forward layer
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, article_embeddings, threat_embeddings):
        # Article transformers flow

        # Cross-correlation between articles and threats
        cross_corr_1 = F.conv1d(article_embeddings.unsqueeze(1), threat_embeddings.flip(-1).unsqueeze(1),
                                padding='same').squeeze(1)
        cross_corr_2 = F.conv1d(threat_embeddings.unsqueeze(1), article_embeddings.flip(-1).unsqueeze(1),
                                padding='same').squeeze(1)

        # Combined transformers flow on cross-correlated sequences
        combined_embeddings = torch.cat((cross_corr_1, cross_corr_2), dim=1)
        for transformer in self.combined_transformers:
            combined_embeddings = transformer(combined_embeddings)

        # Pooling layer
        pooled_output = self.pooling(combined_embeddings.transpose(1, 2)).squeeze(-1)

        # Feed-forward and sigmoid output
        output = self.feed_forward(pooled_output)
        output = torch.sigmoid(output)
        return output
