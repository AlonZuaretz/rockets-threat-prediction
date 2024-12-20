import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_block import TransformerBlock

class ArticlesNN(nn.Module):
    def __init__(self, seq_len, emb_dim=1541, n_heads=4, n_layers=1, hidden_dim=512):
        super(ArticlesNN, self).__init__()
        self.n_layers = n_layers
        self.seq_len = seq_len

        self.projection_layer = nn.Linear(emb_dim, 1544)
        self.positional_encoding = nn.Parameter(torch.randn(seq_len, 1544))
        self.transformers = nn.Sequential(
            TransformerBlock(1544, hidden_dim, n_heads, hidden_dim),
            TransformerBlock(hidden_dim, hidden_dim, n_heads, hidden_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.projection_layer.weight)
        nn.init.constant_(self.projection_layer.bias, 0)
        nn.init.normal_(self.positional_encoding, mean=0, std=0.1)

    def forward(self, x):
        x = self.projection_layer(x)
        x = x + self.positional_encoding.unsqueeze(0)  # Broadcasting for batch
        x = x.permute(1, 0, 2)  # (seq_length, batch_size, embedding_dim)
        x = self.transformers(x)
        x = x.permute(1, 0, 2)  # (batch_size, seq_length, embedding_dim)
        return x


class ThreatsNN(nn.Module):
    def __init__(self, seq_len, input_dim=9, emb_dim=128, n_heads=4, n_layers=1, hidden_dim=512):
        super(ThreatsNN, self).__init__()
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.positional_encoding = nn.Parameter(torch.randn(seq_len, emb_dim))
        self.transformers = nn.Sequential(
            TransformerBlock(emb_dim, hidden_dim, n_heads, hidden_dim),
            TransformerBlock(hidden_dim, hidden_dim, n_heads, hidden_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.constant_(self.embedding.bias, 0)
        nn.init.normal_(self.positional_encoding, mean=0, std=0.1)

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.positional_encoding.unsqueeze(0)  # Broadcasting for batch
        x = x.permute(1, 0, 2)  # (seq_length, batch_size, embedding_dim)
        x = self.transformers(x)
        x = x.permute(1, 0, 2)  # (batch_size, seq_length, embedding_dim)
        return x


class CombinedNN(nn.Module):
    def __init__(self, seq_len1, seq_len2, output_size=4, emb_dim=512, num_heads=4, hidden_dim=512):
        super(CombinedNN, self).__init__()
        self.cross_attention_1 = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, batch_first=True)
        self.cross_attention_2 = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear((seq_len1 + seq_len2) * hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_size)
        )
        self.sigmoid = nn.Sigmoid()

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, seq1, seq2):
        attn_output_1, _ = self.cross_attention_1(query=seq1, key=seq2, value=seq2)
        attn_output_2, _ = self.cross_attention_2(query=seq2, key=seq1, value=seq1)
        concat_output = torch.cat((attn_output_1, attn_output_2), dim=1)
        x = concat_output.view(-1, (seq1.size(1) + seq2.size(1)) * seq1.size(2))
        output = self.fc(x)
        normalized_output = self.sigmoid(output)
        return output, normalized_output
