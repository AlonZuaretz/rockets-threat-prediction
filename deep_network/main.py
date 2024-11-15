import torch
import torch.nn as nn

from data_helper import read_from_csv, preprocess_data, get_embeddings
from separate_NN import ArticlesNN, ThreatsNN
from combined_NN import CombinedNN


def train_model(articles_NN, threats_NN, combined_NN, articles_ds, threats_ds):
    articles_seq_len = 10
    threats_seq_len = 100
    combined_seq_len = 20
    batch_size = 16

    num_epochs = 10
    lr = 1e-4


    articles_NN.train()
    threats_NN.train()
    combined_NN.train()

    for epoch in range(num_epochs):


    return


if __name__ == "__main__":
    # Data Acquisition
    articles_csv_path = 'articles.csv'  # Path to CSV file containing articles (Title, Body, Day, Hour)
    threats_csv_path = 'threats.csv'  # Path to CSV file containing threats data

    articles_df, threats_df = read_from_csv(articles_csv_path, threats_csv_path)

    # Preprocess data
    articles_df_pre, threats_df_pre = preprocess_data(articles_df, threats_df)

    # Get embeddings for each data type:
    article_embeddings, threat_embeddings = get_embeddings(articles_df_pre, threats_df_pre)

    # Get Datasets:
    articles_ds = []
    threats_ds = []

    # Instantiate the models:
    articles_model = ArticlesNN()
    threats_model = ThreatsNN()
    combined_model = CombinedNN()

    # Parameters:


    # Train the model
    train_model(articles_model, threats_model, combined_model, articles_ds, threats_ds)
