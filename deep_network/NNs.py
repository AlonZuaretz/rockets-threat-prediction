import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, seq_length, input_dim=7, emb_dim=128, n_heads=4, n_layers=1, hidden_dim=64):
        super(ThreatsNN, self).__init__()
        self.n_layers = n_layers
        # Embedding layer to convert each element of the sequence into an embedding
        self.embedding = nn.Linear(input_dim, emb_dim)

        # Positional encoding to retain information about the position of each element in the sequence
        self.positional_encoding = nn.Parameter(torch.randn(seq_length, emb_dim))

        self.transformer = TransformerBlock(emb_dim, n_heads, hidden_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        # Embedding each sequence element
        x = self.embedding(x)  # shape: (batch_size, seq_length, embedding_dim)

        # Adding positional encoding
        x = x + self.positional_encoding.unsqueeze(0)  # Broadcasting for batch

        # Permute dimensions for transformer compatibility
        # Expected shape: (seq_length, batch_size, embedding_dim)
        x = x.permute(1, 0, 2)

        # Transformer encoder
        x = self.transformer(x)  # shape: (seq_length, batch_size, embedding_dim)

        # Permute back to original dimension format
        x = x.permute(1, 0, 2)  # shape: (batch_size, seq_length, embedding_dim)

        return x



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


