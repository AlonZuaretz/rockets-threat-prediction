import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_block import TransformerBlock


class ArticlesNN(nn.Module):
    def __init__(self, seq_len, emb_dim=1541, n_heads=1, n_layers=1, hidden_dim=512):
        super(ArticlesNN, self).__init__()
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.positional_encoding = nn.Parameter(torch.randn(seq_len, emb_dim))
        self.transformer = TransformerBlock(emb_dim, hidden_dim, n_heads, hidden_dim)


    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
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


class ThreatsNN(nn.Module):
    def __init__(self, seq_len, input_dim=9, emb_dim=128, n_heads=4, n_layers=1, hidden_dim=512):
        super(ThreatsNN, self).__init__()
        self.n_layers = n_layers
        self.seq_len = seq_len
        # Embedding layer to convert each element of the sequence into an embedding
        self.embedding = nn.Linear(input_dim, emb_dim)
        # Positional encoding to retain information about the position of each element in the sequence
        self.positional_encoding = nn.Parameter(torch.randn(seq_len, emb_dim))
        self.transformer = TransformerBlock(emb_dim, hidden_dim, n_heads, hidden_dim)

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
    def __init__(self, seq_len1, seq_len2, output_size=4, emb_dim=512, num_heads=4, hidden_dim=512):
        super(CombinedNN, self).__init__()
        self.cross_attention_1 = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, batch_first=True)
        self.cross_attention_2 = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, batch_first=True)

        # Linear layers for further processing after concatenation
        self.linear1 = nn.Linear((seq_len1+seq_len2) * hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_size)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, seq1, seq2):
        # seq1: (batch_size, sequence_length_1, embed_dim)
        # seq2: (batch_size, sequence_length_2, embed_dim)

        # Cross attention: seq1 attends to seq2 and seq2 attends to seq1
        attn_output_1, _ = self.cross_attention_1(query=seq1, key=seq2, value=seq2)
        attn_output_2, _ = self.cross_attention_2(query=seq2, key=seq1, value=seq1)

        # Concatenate the outputs of both attention layers along the embedding dimension
        concat_output = torch.cat((attn_output_1, attn_output_2),
                                  dim=1)  # (batch_size, sequence_length_1 + sequence_length_2, embed_dim )

        x = concat_output.view(-1, (seq1.size(1) + seq2.size(1)) * seq1.size(2))

        # Pass through linear layers
        x = self.linear1(x)  # (batch_size, sequence_length_1, hidden_dim)
        x = self.relu(x)
        x = self.linear2(x)  # (batch_size, sequence_length_1, output_size)

        # Apply sigmoid activation to the final output
        output = self.sigmoid(x)  # (batch_size, sequence_length_1, output_size)

        return output


