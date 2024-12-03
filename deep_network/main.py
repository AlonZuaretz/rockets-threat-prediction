import torch

from data_handling.process_data import process
from data_handling.read import read_from_csv
from NNs import ArticlesNN, ThreatsNN, CombinedNN
from train import train_model


if __name__ == "__main__":

    articles_df, threats_df, num_locations = read_from_csv()
    print(num_locations)

    # Set hyper-parameters
    articles_seqlen = 20
    threats_seqlen = 60
    batch_size = 16
    num_epochs = 100
    lr = 1e-3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Process data
    train_dl, val_dl, test_dl = process(articles_df, threats_df, articles_seqlen, threats_seqlen, batch_size)

    # Instantiate the models:
    articles_model = ArticlesNN(articles_seqlen).to(device)
    threats_model = ThreatsNN(threats_seqlen, input_dim=5 + num_locations).to(device)
    combined_model = CombinedNN(articles_seqlen, threats_seqlen, output_size=num_locations).to(device)

    # Train the model
    train_model(articles_model, threats_model, combined_model, train_dl, val_dl, device, num_epochs, lr)
