import torch
import torch.nn as nn


class ArticlesNN(nn.Module):
    def __init__(self, seq_len, emb_dim=1541, hidden_dim=512, num_layers=1):
        super(ArticlesNN, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.projection_layer = nn.Linear(emb_dim, hidden_dim)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        # Adding projection layer
        x = self.projection_layer(x)

        # LSTM layer
        x, _ = self.lstm(x)  # shape: (batch_size, seq_length, hidden_dim)

        return x


class ThreatsNN(nn.Module):
    def __init__(self, seq_len, input_dim=9, emb_dim=128, hidden_dim=512, num_layers=1):
        super(ThreatsNN, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding layer to convert each element of the sequence into an embedding
        self.embedding = nn.Linear(input_dim, emb_dim)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        # Embedding each sequence element
        x = self.embedding(x)  # shape: (batch_size, seq_length, embedding_dim)

        # LSTM layer
        x, _ = self.lstm(x)  # shape: (batch_size, seq_length, hidden_dim)

        return x


class CombinedNN(nn.Module):
    def __init__(self, seq_len1, seq_len2, output_size=4, emb_dim=1024, num_heads=4, hidden_dim=512):
        super(CombinedNN, self).__init__()
        self.cross_attention_1 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.cross_attention_2 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # Linear layers for further processing after concatenation
        self.fc = nn.Sequential(
            nn.Linear((seq_len1 + seq_len2) * hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size)
        )

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
        output = self.fc(x)  # (batch_size, sequence_length_1, hidden_dim)
        normalized_output = self.sigmoid(output)

        return output, normalized_output
