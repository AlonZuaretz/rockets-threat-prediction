import torch
import pandas as pd
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer
from NetTab import NetTab


def preprocess_data(articles_df, threats_df):
    # Perform any normalizations if needed
    # For now:
    return articles_df, threats_df


def read_from_csv(articles_csv_path, threats_csv_path):
    threats_df = pd.read_csv(threats_csv_path)
    articles_df = pd.read_csv(articles_csv_path)
    return articles_df, threats_df


def get_embeddings(articles_df, threats_df):
    article_embedding_model = ArticleEmbeddingNet()
    article_embeddings = article_embedding_model.get_embeddings_from_df(
        articles_df)  # Shape: (N, article_embedding_size)

    # Generate tabular embeddings from threats inputs
    threats_inputs = torch.tensor(threats_df.values, dtype=torch.float32)  # Assuming threats are all numeric data
    threat_embedding_model = ThreatsEmbeddingNet()
    threat_embeddings = threat_embedding_model(threats_inputs)  # Shape: (M, threats_embedding_size)

    return article_embeddings, threat_embeddings


class ArticleEmbeddingNet(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(ArticleEmbeddingNet, self).__init__()
        # Load a pretrained large language model
        self.llm = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_embeddings_from_df(self, df):
        # Load CSV and extract Title, Body, Day, and Hour
        titles = df['Title'].tolist()
        bodies = df['Body'].tolist()
        day_hour_data = df[['Day', 'Hour']].values

        # Prepare inputs for the LLM
        inputs = self.tokenizer(titles, bodies, padding=True, truncation=True, return_tensors="pt")

        # Extract embeddings
        with torch.no_grad():
            text_embeddings = self.llm(**inputs).last_hidden_state[:, 0, :]  # Assuming CLS token is used

        # Convert day and hour to tensor and concatenate with text embeddings
        day_hour_tensor = torch.tensor(day_hour_data, dtype=torch.float32)
        combined_embeddings = torch.cat((text_embeddings, day_hour_tensor), dim=1)
        return combined_embeddings


class ThreatsEmbeddingNet(nn.Module):
    def __init__(self, input_dim=770, embedding_dim=256):  # Updated input_dim to match concatenated size
        super(ThreatsEmbeddingNet, self).__init__()
        # Load NetTab
        self.net_tab = NetTab(input_dim=input_dim, output_dim=embedding_dim)

    def forward(self, inputs):
        # Extract embeddings from tabular data
        embeddings = self.net_tab(inputs)
        return embeddings