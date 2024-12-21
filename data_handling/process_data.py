import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from transformers import BertModel, BertTokenizerFast
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from datetime import datetime

class ArticleEmbeddingNet(nn.Module):
    def __init__(self, model_name='onlplab/alephbert-base'):
        super(ArticleEmbeddingNet, self).__init__()
        # Load a pretrained large language model
        self.model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model.eval()

    def forward(self, df):
        # Load CSV and extract Title, Body, Day, and Hour
        titles = df['Main_Titles'].tolist()
        bodies = df['Sub_Titles'].tolist()

        # Prepare inputs for the LLM for titles
        title_inputs = self.tokenizer(titles, padding=True, truncation=True, return_tensors="pt")
        body_inputs = self.tokenizer(bodies, padding=True, truncation=True, return_tensors="pt")

        # Extract embeddings for titles
        with torch.no_grad():
            title_embeddings = self.model(**title_inputs).last_hidden_state[:, 0, :]  # CLS token representation

        # Extract embeddings for bodies
        with torch.no_grad():
            body_embeddings = self.model(**body_inputs).last_hidden_state[:, 0, :]  # CLS token representation

        # Update DataFrame with embeddings
        df['Main_Titles'] = title_embeddings.tolist()
        df['Sub_Titles'] = body_embeddings.tolist()

        return df


class CreateDataSet(Dataset):
    def __init__(self, ds1, ds2, labels, last_time, seq_len1, seq_len2, time_resolution):
        self.ds1 = ds1  # Articles
        self.ds2 = ds2  # Threats
        self.labels = labels
        self.last_time = last_time
        self.seq_len1 = seq_len1
        self.seq_len2 = seq_len2
        self.time_resolution = time_resolution
        self.cache = {}  # Cache to store the index of the last match

    def __len__(self):
        return len(self.ds2)

    def __getitem__(self, idx):
        # Get the sequence from the second dataset based on its sequence length
        seq2 = self.ds2[idx][:][:]

        # Find the last timestamp of the current sequence from dataset1
        last_time = self.last_time[idx]  # Assuming time is the first element in each sample

        # Find the corresponding sequence in the first dataset ending with last_time
        seq1 = self.find_corresponding_sequence(last_time)
        if seq1.shape[0] == 0:
            x=5


        label = self.labels[idx, :]

        return seq1, seq2, label

    def find_corresponding_sequence(self, last_time):
        # Check if the last_time is already in the cache
        if last_time in self.cache:
            idx = self.cache[last_time]
        else:
            # Find the sequence in dataset1 that ends with the given time
            idx = None
            for (i, time) in enumerate(self.ds1[:, 0]):
                if time > last_time + self.time_resolution*3600:
                    idx = i
                    break

            if idx is None:
                raise ValueError(f"No corresponding sequence found in dataset1 for time {last_time}")

            # Cache the index for future use
            self.cache[last_time] = idx

        # Return the sequence from dataset1 based on the cached or newly found index
        return self.ds1[idx - self.seq_len1:idx, 1:]


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

def one_hot_encoder(df, time_resolution):
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
            'week day': group['week day'].iloc[0],
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
        next_time_id = time_id + pd.Timedelta(hours=time_resolution)
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
    time_id_df = labels_df['time_id']
    labels_df = labels_df.drop(columns=['time_id'])

    return result_df, labels_df, time_id_df


def process(articles_df, threats_df, articles_seqlen, threats_seqlen, batch_size, time_resolution):

    # Pass articles through an LLM to get embeddings:
    if articles_df:
        article_embedding_model = ArticleEmbeddingNet()
        articles_df = article_embedding_model(articles_df)  # Shape: (N, article_embedding_size)
    else:
        articles_df = pd.read_csv(r"data\articles\embedded_articles_alephbert.csv")

    # get unique row for each hour and date with a one-hot enconding:
    threats_df, labels_df, time_id_df = one_hot_encoder(threats_df, time_resolution)

    articles_np = articles_df.to_numpy()
    threats_np, labels_np = threats_df.to_numpy(), labels_df.to_numpy()

    # Min Max normalize each column separately
    articles_np[:, 2:7] = min_max_normalize(articles_np[:, 2:7])
    threats_np = min_max_normalize(threats_np)

    # Split the embedded strings into long rows:
    main_titles = articles_np[:, 7]
    sub_titles = articles_np[:, 8]
    split_rows = [list(map(float, row.strip('[]').split(','))) for row in main_titles]
    main_titles = np.array(split_rows, dtype=np.float32)
    split_rows = [list(map(float, row.strip('[]').split(','))) for row in sub_titles]
    sub_titles = np.array(split_rows, dtype=np.float32)

    # Concatenate the articles array with the main and sub titles
    articles_np = articles_np[:, 1:7]
    articles_np = np.array(articles_np, dtype=np.float32)
    articles_np = np.concatenate((articles_np, main_titles, sub_titles), axis=1)

    # Split the threats data into sequences
    num_sequences = len(threats_np) - threats_seqlen + 1
    threats_dim = threats_np.shape[1]
    labels_dim = labels_np.shape[1]
    threats_sequences = np.zeros((num_sequences, threats_seqlen, threats_dim), dtype=np.float32)
    labels = np.zeros((num_sequences, labels_dim), dtype=np.float32)
    last_time = np.zeros(num_sequences)

    for i in range(0, num_sequences, 1):
        threats_sequences[i, :, :] = threats_np[i:i+threats_seqlen, :]
        labels[i, :] = labels_np[i+threats_seqlen-1, :]
        last_time[i] = int(time_id_df[i+threats_seqlen-1].timestamp())

    # Split threats into train, test, validation sets
    threats_train, threats_test, labels_train, labels_test, last_time_train, last_time_test = train_test_split(
        threats_sequences, labels, last_time, test_size=0.3, random_state=42
    )

    # Dataset:
    train_ds = CreateDataSet(articles_np, threats_train, labels_train, last_time_train, articles_seqlen, threats_seqlen, time_resolution)
    test_ds = CreateDataSet(articles_np, threats_test, labels_test, last_time_train, articles_seqlen, threats_seqlen, time_resolution)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_dl, test_dl




