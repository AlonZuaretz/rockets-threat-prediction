import torch

from data_handling.process_data import process
from data_handling.read import read_from_csv
from NNs import ArticlesNN, ThreatsNN, CombinedNN


def train_model(articles_NN, threats_NN, combined_NN, articles_dl, threats_dl, device, num_epochs, lr):

    articles_NN.train()
    threats_NN.train()
    combined_NN.train()

    for epoch in range(num_epochs):
        for (threats_seq, index) in threats_dl:
            threats_seq = threats_seq.to(device)
            output = threats_NN(threats_seq)


    return


if __name__ == "__main__":
    # Data Acquisition
    articles_csv_path = 'articles.csv'  # Path to CSV file containing articles (Title, Body, Day, Hour)
    threats_csv_path = 'threats.csv'  # Path to CSV file containing threats data

    articles_df, threats_df = read_from_csv()
    # Set hyper-parameters
    articles_seqlen = 20
    threats_seqlen = 30
    batch_size = 8
    num_epochs = 10
    lr = 1e-4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Process data
    articles_train_dl, articles_val_dl, articles_test_dl, threats_train_dl, threats_val_dl, threats_test_dl =\
        process(articles_df, threats_df, articles_seqlen, threats_seqlen, batch_size)

    # Instantiate the models:
    articles_model = ArticlesNN().to(device)
    threats_model = ThreatsNN(threats_seqlen).to(device)
    combined_model = CombinedNN().to(device)

    # Train the model
    train_model(articles_model, threats_model, combined_model, articles_train_dl, threats_train_dl, device, num_epochs, lr)
