import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.io
import os
import torch.nn.functional as F
import sys

from torch.optim.lr_scheduler import StepLR
from pynput import keyboard

stop_training = False


def calc_pos_weight(dl_train):
    _, _, labels = next(iter(dl_train))  # Extract a single batch to determine shape
    num_locations = labels.shape[1]  # Determine the number of locations (channels)
    pos_weights = []

    for location in range(num_locations):
        num_ones = 0
        num_zeros = 0
        for (_, _, labels) in dl_train:
            num_ones += (labels[:, location] == 1).sum().item()
            num_zeros += (labels[:, location] == 0).sum().item()

        if num_ones == 0:
            pos_weights.append(1.0)  # Prevent division by zero
        else:
            pos_weights.append(num_zeros / num_ones)

    print(f"Positional weights for locations: {pos_weights}")
    return torch.tensor(pos_weights, dtype=torch.float)



def on_press(key):
    global stop_training
    try:
        if key == keyboard.Key.esc:
            stop_training = True
            print("\nTraining interrupted by user.")
            return False
    except Exception as e:
        print(f"Error: {e}")


def train_model(articles_NN, threats_NN, combined_NN, dl_train, dl_test, device, num_epochs, lr, load_run=None):
    global stop_training

    # Load models from a specific run if specified
    if load_run is not None:
        run_folder = f'results/run_{load_run}'
        if os.path.exists(run_folder):
            articles_NN.load_state_dict(torch.load(os.path.join(run_folder, 'best_model.pth'))['articles_NN'])
            threats_NN.load_state_dict(torch.load(os.path.join(run_folder, 'best_model.pth'))['threats_NN'])
            combined_NN.load_state_dict(torch.load(os.path.join(run_folder, 'best_model.pth'))['combined_NN'])
            print(f"Loaded models from run {load_run}")
        else:
            print(f"Run {load_run} not found. Starting training from scratch.")

    articles_NN.train()
    threats_NN.train()
    combined_NN.train()

    pos_weight = calc_pos_weight(dl_train).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(list(articles_NN.parameters()) + list(threats_NN.parameters()) + list(combined_NN.parameters()), lr=lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    train_loss_list = []
    train_mae_list = []
    test_loss_list = []
    test_mae_list = []
    test_avg_class_list = []
    test_true_positive_list = []

    # Create results directory if it doesn't exist
    base_path = 'results'

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

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    num_of_ones = torch.zeros(1)
    num_of_zeros = torch.zeros(1)

    date_mat = torch.zeros((len(dl_test.dataset), 5))
    k = 0
    batch_size = 16
    for articles_seq, threats_seq, labels in dl_test:
        num_of_ones += labels.sum()
        num_of_zeros += (labels.shape[0] * labels.shape[1] - labels.sum())
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



    for epoch in range(num_epochs):
        if stop_training:
            break

        start_time = time.time()
        epoch_loss = 0.0
        epoch_mae_loss = 0.0
        num_batches = 0

        train_outputs = []
        train_labels = []

        for (articles_seq, threats_seq, labels) in dl_train:
            threats_seq = threats_seq.to(device)
            articles_seq = articles_seq.to(device)
            labels = labels.to(device)

            articles_output = articles_NN(articles_seq)
            threats_output = threats_NN(threats_seq)

            outputs, normalized_outputs = combined_NN(articles_output, threats_output)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            mae_loss = torch.mean(torch.abs(normalized_outputs - labels))
            epoch_mae_loss += mae_loss.item()
            num_batches += 1

            train_outputs.extend(normalized_outputs.detach().cpu().numpy())
            train_labels.extend(labels.detach().cpu().numpy())

        train_loss = epoch_loss / num_batches
        train_mae_loss = epoch_mae_loss / num_batches
        train_loss_list.append(train_loss)
        train_mae_list.append(train_mae_loss)

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"Epoch [{epoch + 1}] Training score:\n"
              f" Average Loss: {train_loss:.4f}\n"
              f" MAE: {train_mae_loss:.4f}\n"
              f" Time: {epoch_duration:.2f} seconds")

        # Validation every epoch
        if (epoch + 1) % 1 == 0:
            articles_NN.eval()
            threats_NN.eval()
            combined_NN.eval()
            epoch_loss = 0.0
            epoch_mae_loss = 0.0
            test_class_correct = 0
            test_total = 0
            test_true_positive = 0
            test_true_negative = 0
            test_batches = 0

            test_outputs = []
            test_labels = []

            with torch.no_grad():
                for articles_seq, threats_seq, labels in dl_test:
                    threats_seq = threats_seq.to(device)
                    articles_seq = articles_seq.to(device)
                    labels = labels.to(device)

                    articles_output = articles_NN(articles_seq)
                    threats_output = threats_NN(threats_seq)

                    outputs, normalized_outputs = combined_NN(articles_output, threats_output)

                    loss = criterion(outputs, labels)
                    epoch_loss += loss.item()

                    # Calculate average difference loss (MAE)
                    mae_loss = torch.mean(torch.abs(normalized_outputs - labels))
                    epoch_mae_loss += mae_loss.item()

                    # Classify outputs around 0.5
                    predicted_classes = (normalized_outputs >= 0.5).float()
                    test_class_correct += (predicted_classes == labels).sum().item()
                    test_total += labels.numel()

                    # Calculate true positives (how many of the ones in the labels were correctly predicted)
                    test_true_positive += ((predicted_classes == 1) & (labels == 1)).sum().item()
                    test_true_negative += ((predicted_classes == 0) & (labels == 0)).sum().item()
                    test_batches += 1

                    test_outputs.extend(normalized_outputs.detach().cpu().numpy())
                    test_labels.extend(labels.detach().cpu().numpy())

            test_loss = epoch_loss / test_batches
            test_mae_loss = epoch_mae_loss / test_batches
            test_accuracy = test_class_correct / test_total

            test_loss_list.append(test_loss)
            test_mae_list.append(test_mae_loss)
            test_avg_class_list.append(test_accuracy)
            test_true_positive_list.append(test_true_positive)

            print(f"Epoch [{epoch + 1}] Validation score:\n"
                  f" Validation Loss  {test_loss:.4f}, MAE: {test_mae_loss:.4f}, \n"
                  f" Classification Accuracy: {test_accuracy:.4f}\n"
                  f" True Positives out of all positives: {test_true_positive} / {num_of_ones.item()} \n"
                  f" True Negatives out of all negatives: {test_true_negative} / {num_of_zeros.item()} \n\n")

            articles_NN.train()
            threats_NN.train()
            combined_NN.train()

        # Step the scheduler
        scheduler.step()

    # Save the last model
    last_model_state = {
        'articles_NN': articles_NN.state_dict(),
        'threats_NN': threats_NN.state_dict(),
        'combined_NN': combined_NN.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch + 1  # Save the current epoch
    }
    torch.save(last_model_state, os.path.join(result_folder, 'last_model.pth'))
    print(f"Last model saved as '{os.path.join(result_folder, 'last_model.pth')}'")

    # Save losses to .mat file
    scipy.io.savemat(os.path.join(result_folder, 'loss_data.mat'), {
        'train_loss_list': train_loss_list,
        'train_mae_list': train_mae_list,
        'test_loss_list': test_loss_list,
        'test_avg_diff': test_mae_list,
        'test_avg_class': test_avg_class_list,
        'test_true_positive': test_true_positive_list
    })
    print(f"Loss data saved to '{os.path.join(result_folder, 'loss_data.mat')}'")

    # Save outputs vs labels to .mat file
    scipy.io.savemat(os.path.join(result_folder, 'outputs_vs_labels_train.mat'), {
        'train_outputs': train_outputs,
        'train_labels': train_labels
    })
    print(f"Outputs vs labels saved to '{os.path.join(result_folder, 'outputs_vs_labels_train.mat')}'")

    # Save validation outputs vs labels to .mat file
    scipy.io.savemat(os.path.join(result_folder, f'outputs_vs_labels_val.mat'), {
        'test_outputs': test_outputs,
        'test_labels': test_labels,
        'test_dates': date_mat
    })
    print(f"Validation outputs vs labels saved to '{os.path.join(result_folder, f'outputs_vs_labels_val.mat')}'")

    print("Training complete.")
    return
