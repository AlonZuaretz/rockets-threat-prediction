import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split



class ArticleEmbeddingNet(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(ArticleEmbeddingNet, self).__init__()
        # Load a pretrained large language model
        self.llm = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, df):
        # Load CSV and extract Title, Body, Day, and Hour
        titles = df['Title'].tolist()
        bodies = df['Body'].tolist()

        # Prepare inputs for the LLM for titles
        title_inputs = self.tokenizer(titles, padding=True, truncation=True, return_tensors="pt")
        body_inputs = self.tokenizer(bodies, padding=True, truncation=True, return_tensors="pt")

        # Extract embeddings for titles
        with torch.no_grad():
            title_embeddings = self.llm(**title_inputs).last_hidden_state[:, 0, :]  # CLS token representation

        # Extract embeddings for bodies
        with torch.no_grad():
            body_embeddings = self.llm(**body_inputs).last_hidden_state[:, 0, :]  # CLS token representation

        # Update DataFrame with embeddings
        df['Title'] = title_embeddings.tolist()
        df['Body'] = body_embeddings.tolist()

        return df


class ThreatsDataset(Dataset):
    def __init__(self, data, labels, sequence_length):
        self.data = data
        self.labels = labels
        self.sequence_length = sequence_length
        self.num_rows = data.shape[0]

    def __len__(self):
        return self.num_rows - self.sequence_length + 1

    def __getitem__(self, idx):
        sequence = self.data[idx:idx + self.sequence_length, :]
        label = self.labels[idx, :]
        return sequence, label, idx


class ArticlesDataset(Dataset):
    def __init__(self, data, df, sequence_length):
        self.data = data
        self.df = df  # DataFrame with 'date' and 'hour' columns
        self.sequence_length = sequence_length
        self.num_rows = data.shape[0]

    def __len__(self):
        return self.num_rows - self.sequence_length + 1

    def find_start_index(self, date, hour):
        # Find the first index where the date and hour are greater than the given input
        mask = (self.df['date'] < date) | ((self.df['date'] == date) & (self.df['hour'] < hour))
        indices = self.df[mask].index
        if len(indices) == 0:
            raise ValueError("No valid index found before the given date and hour.")
        return indices[-1]

    def __getitem__(self, idx):
        date, hour = idx
        start_idx = self.find_start_index(date, hour)
        if start_idx - self.sequence_length + 1 < 0:
            raise ValueError("Not enough samples prior to the given date and hour for the specified sequence length.")
        sequence = self.data[start_idx - self.sequence_length + 1:start_idx + 1, :]
        return sequence, start_idx - self.sequence_length + 1


class CreateDataSet(Dataset):
    def __init__(self, ds1, ds2, seq_len1, seq_len2):
        self.ds1 = ds1
        self.ds2 = ds2
        self.seq_len1 = seq_len1
        self.seq_len2 = seq_len2

        def __len__(self):
            return len(self.dataset1)

        def __getitem__(self, idx):
            # Get the sequence from the first dataset based on its sequence length
            seq1 = self.dataset1[idx][:self.seq_len1]

            # Find the last timestamp of the current sequence from dataset1
            last_time = seq1[-1][0]  # Assuming time is the first element in each sample

            # Find the corresponding sequence in the second dataset ending with last_time
            seq2 = self.find_corresponding_sequence(last_time)

            # Adjust the sequence length for dataset2
            seq2 = seq2[-self.seq_len2:]

            return seq1, seq2

        def find_corresponding_sequence(self, last_time):
            # Find the sequence in dataset2 that ends with the given time
            for seq in self.dataset2:
                if seq[-1][0] == last_time:
                    return seq
            raise ValueError(f"No corresponding sequence found in dataset2 for time {last_time}")

def min_max_normalize(array):

    if array.size == 0:
        print("The array is empty. Returning an empty array.")
        return array

    # Create an empty array with the same shape to store normalized values
    normalized_array = np.zeros_like(array, dtype=np.float32)

    # Iterate over each column
    for col in range(array.shape[1]):
        min_val = array[:, col].min()
        max_val = array[:, col].max()

        if max_val == min_val:
            # If all values in the column are the same, set normalized values to 0.5
            normalized_array[:, col] = 0.5
        else:
            # Apply min-max normalization
            normalized_array[:, col] = (array[:, col] - min_val) / (max_val - min_val)

    return normalized_array

def one_hot_encoder(df):
    # Create a unique time-based identifier for grouping

    df['time_id'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])

    # Find unique locations
    unique_locations = sorted(df['location'].unique())

    # Map locations to indices
    location_to_index = {loc: i for i, loc in enumerate(unique_locations)}

    # Group by the unique time identifier
    grouped = df.groupby('time_id')

    # Build the result DataFrame
    result = []
    for time_id, group in grouped:
        # Initialize a zero vector for one-hot encoding
        one_hot_vector = np.zeros(len(unique_locations), dtype=int)
        # Set 1 for each unique location in the group
        for loc in group['location'].unique():
            one_hot_vector[location_to_index[loc]] = 1
        # Add the row to the result
        row = {
            'hour': group['hour'].iloc[0],
            'day': group['day'].iloc[0],
            'month': group['month'].iloc[0],
            'year': group['year'].iloc[0]
        }
        # Add one-hot vector as columns
        row.update({f'loc_{loc}': val for loc, val in zip(unique_locations, one_hot_vector)})
        result.append(row)

    # Convert result to DataFrame
    result_df = pd.DataFrame(result)

    # Create a shifted DataFrame for labels
    labels = []
    for time_id, group in grouped:
        next_time_id = time_id + pd.Timedelta(hours=1)
        # Check if next_time_id exists
        if next_time_id in grouped.groups:
            next_group = grouped.get_group(next_time_id)
            # Create a one-hot vector for the labels
            one_hot_vector = np.zeros(len(unique_locations), dtype=int)
            for loc in next_group['location'].unique():
                one_hot_vector[location_to_index[loc]] = 1
        else:
            # No next hour: all zeros
            one_hot_vector = np.zeros(len(unique_locations), dtype=int)
        # Add the label row
        label_row = {'time_id': time_id}
        label_row.update({f'label_loc_{loc}': val for loc, val in zip(unique_locations, one_hot_vector)})
        labels.append(label_row)

    # Convert labels to DataFrame
    labels_df = pd.DataFrame(labels)

    return result_df, labels_df



def process(articles_df, threats_df, articles_seqlen, threats_seqlen, batch_size):

    # Pass articles through an LLM to get embeddings:
    # article_embedding_model = ArticleEmbeddingNet()
    # articles_df = article_embedding_model(articles_df)  # Shape: (N, article_embedding_size)

    # get unique row for each hour and date with a one-hot enconding:
    threats_df, labels_df = one_hot_encoder(threats_df)

    # articles_np = articles_df.to_numpy()
    threats_np, labels_np = threats_df.to_numpy(), labels_df.to_numpy()

    # articles_np = min_max_normalize(articles_np)
    threats_np = min_max_normalize(threats_np)

    # Split articles into train, test, validation sets
    # articles_train, articles_temp = train_test_split(articles_np, test_size=0.3, random_state=42)
    # articles_val, articles_test = train_test_split(articles_temp, test_size=(1 / 3), random_state=42)

    # Split threats into train, test, validation sets
    threats_train, threats_temp, labels_train, labels_temp = train_test_split(threats_np, labels_np, test_size=0.3, random_state=42)
    threats_val, threats_test, labels_val, labels_test = train_test_split(threats_temp, labels_temp, test_size=(1 / 3), random_state=42)


    # Articles Dataloader:
    # articles_train_ds = ArticlesDataset(articles_train, threats_df[['date', 'hour']], articles_seqlen)
    # articles_val_ds = ArticlesDataset(articles_val,threats_df[['date', 'hour']], articles_seqlen)
    # articles_test_ds = ArticlesDataset(articles_test, threats_df[['date', 'hour']], articles_seqlen)

    # articles_train_dl = DataLoader(articles_train_ds, batch_size=batch_size, shuffle=True)
    # articles_val_dl = DataLoader(articles_val_ds, batch_size=batch_size, shuffle=False)
    # articles_test_dl = DataLoader(articles_test_ds, batch_size=batch_size, shuffle=False)

    articles_train_dl = []
    articles_val_dl = []
    articles_test_dl = []

    # Threats Dataloader:
    threats_train_ds = ThreatsDataset(threats_train, labels_train, threats_seqlen)
    threats_val_ds = ThreatsDataset(threats_val, labels_val, threats_seqlen)
    threats_test_ds = ThreatsDataset(threats_test, labels_test, threats_seqlen)

    threats_train_dl = DataLoader(threats_train_ds, batch_size=batch_size, shuffle=True)
    threats_val_dl = DataLoader(threats_val_ds, batch_size=batch_size, shuffle=False)
    threats_test_dl = DataLoader(threats_test_ds, batch_size=batch_size, shuffle=False)

    return articles_train_dl, articles_val_dl, articles_test_dl, threats_train_dl, threats_val_dl, threats_test_dl




