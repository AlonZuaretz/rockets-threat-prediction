import torch
import json

from data_handling.process_data import process
from data_handling.read import read_from_csv
from deep_network.NNs_LSTM import ArticlesNN, ThreatsNN, CombinedNN
from deep_network.train import train_model

if __name__ == "__main__":

    read_embedded_articles = True
    time_resolution = 6
    articles_df, threats_df, locations_mapping = read_from_csv(read_embedded_articles, time_resolution)
    num_locations = len(locations_mapping)

    # To save the locations mappings:
    # with open(r"locations_dict.txt", "w", encoding="utf-8") as f:
    #     f.write(str(locations_mapping))

    # Set hyper-parameters
    articles_seqlen = 20
    threats_seqlen = 30
    batch_size = 16
    num_epochs = 30
    lr = 1e-3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Process data
    train_dl, test_dl = process(articles_df, threats_df,
                                        articles_seqlen, threats_seqlen, batch_size, time_resolution)

    # Instantiate the models:
    articles_model = ArticlesNN(articles_seqlen, hidden_dim=1024).to(device)
    threats_model = ThreatsNN(threats_seqlen, input_dim=5 + num_locations, hidden_dim=1024).to(device)
    combined_model = CombinedNN(articles_seqlen, threats_seqlen, hidden_dim=1024, output_size=num_locations).to(device)

    # Train the model
    train_model(articles_model, threats_model, combined_model, train_dl, test_dl, device, num_epochs, lr)
