import torch
import os
import scipy

from evaluate.process_data_eval import process
from evaluate.read_eval import read_from_csv
from deep_network.NNs_LSTM import ArticlesNN, ThreatsNN, CombinedNN

def find_dates(dl):
    date_mat = torch.zeros((len(dl.dataset), 5))
    k = 0
    batch_size = 16
    for articles_seq, threats_seq, labels in dl:
        for j in range(5):
            for i in range(len(threats_seq)):
                idx = k * batch_size + i
                date_mat[idx, j] = threats_seq[i, -1, j]
        k += 1
    date_mat[:,0] = date_mat[:,0] * 6 + 1
    date_mat[:,1] = date_mat[:,1] * 18
    date_mat[:,2] = date_mat[:,2] * 30 + 1
    date_mat[:,3] = date_mat[:,3] * 11 + 1
    date_mat[:,4] = date_mat[:,4] + 2023

    return date_mat

def eval_model(articles_NN, threats_NN, combined_NN, dl, device):

    date_mat = find_dates(dl)

    # Create results directory if it doesn't exist
    base_path = 'evaluate/results'
    # Ensure the base directory exists
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    # Find the next available folder number
    folder_num = 1
    while os.path.exists(os.path.join(base_path, f'run_{folder_num}')):
        folder_num += 1
    # Create the result folder
    result_folder = os.path.join(base_path, f'run_{folder_num}')
    os.makedirs(result_folder)

    articles_NN.eval()
    threats_NN.eval()
    combined_NN.eval()
    epoch_loss = 0.0
    epoch_mae_loss = 0.0
    test_class_correct = 0
    test_total = 0
    test_true_positive = 0
    test_true_negative = 0
    test_false_positive = 0
    test_false_negative = 0
    test_batches = 0
    test_outputs = []
    test_labels = []

    for articles_seq, threats_seq, labels in dl:
        threats_seq = threats_seq.to(device)
        articles_seq = articles_seq.to(device)
        labels = labels.to(device)

        articles_output = articles_NN(articles_seq)
        threats_output = threats_NN(threats_seq)

        outputs, normalized_outputs = combined_NN(articles_output, threats_output)

        # Classify outputs around 0.5
        predicted_classes = (normalized_outputs >= 0.5).float()
        test_class_correct += (predicted_classes == labels).sum().item()
        test_total += labels.numel()

        # True Positives, False Positives, False Negatives
        test_true_positive += ((predicted_classes == 1) & (labels == 1)).sum().item()
        test_false_positive += ((predicted_classes == 1) & (labels == 0)).sum().item()
        test_false_negative += ((predicted_classes == 0) & (labels == 1)).sum().item()
        test_true_negative += ((predicted_classes == 0) & (labels == 0)).sum().item()

        test_batches += 1

        test_outputs.extend(normalized_outputs.detach().cpu().numpy())
        test_labels.extend(labels.detach().cpu().numpy())

    # Calculate losses and accuracy
    test_mae_loss = epoch_mae_loss / test_batches
    test_loss = epoch_loss / test_batches

    # Calculate Precision, Recall, F1 Score
    precision = test_true_positive / (test_true_positive + test_false_positive + 1e-8)
    recall = test_true_positive / (test_true_positive + test_false_negative + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    print(f"Test score:\n"
          f" Loss: {test_loss:.4f}, MAE: {test_mae_loss:.4f}\n"
          f" Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}\n"
          )

    # Save test outputs vs labels to .mat file
    scipy.io.savemat(os.path.join(result_folder, f'outputs_vs_labels_test.mat'), {
        'test_outputs': test_outputs,
        'test_labels': test_labels,
        'test_dates': date_mat
    })
    print(f"Test outputs vs labels saved to '{os.path.join(result_folder, f'outputs_vs_labels_test.mat')}'")

    return


def evaluate(model_path, alerts_path, articles_dir):
    read_embedded_articles = True
    time_resolution = 6
    articles_df, threats_df, locations_mapping = read_from_csv(
                                                               alerts_path, articles_dir, time_resolution)
    num_locations = len(locations_mapping)
    # To save the locations mappings:
    with open(r"evaluate/locations_dict.txt", "w", encoding="utf-8") as f:
        f.write(str(locations_mapping))

    # Set hyper-parameters
    articles_seqlen = 20
    threats_seqlen = 30
    batch_size = 16
    hidde_dim = 1024

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Process data
    dl = process(articles_df, threats_df,articles_seqlen, threats_seqlen, batch_size, time_resolution)

    # Instantiate the models:
    articles_model = ArticlesNN(articles_seqlen, hidden_dim=hidde_dim).to(device)
    threats_model = ThreatsNN(threats_seqlen, input_dim=5 + num_locations, hidden_dim=hidde_dim).to(device)
    combined_model = CombinedNN(articles_seqlen, threats_seqlen, hidden_dim=hidde_dim, output_size=num_locations).to(device)

    if os.path.exists(model_path):
        articles_model.load_state_dict(torch.load(model_path)['articles_NN'])
        threats_model.load_state_dict(torch.load(model_path)['threats_NN'])
        combined_model.load_state_dict(torch.load(model_path)['combined_NN'])
        print(f"Loaded models from run {model_path}")

    # Evaluate the data
    eval_model(articles_model, threats_model, combined_model, dl, device)
