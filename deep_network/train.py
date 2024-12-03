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
    num_ones = 0
    num_zeros = 0
    for (_, _, labels) in dl_train:
        num_ones += (labels == 1).sum().item()
        num_zeros += (labels == 0).sum().item()

    pos_weight = num_zeros / num_ones
    print(pos_weight)
    return torch.tensor([pos_weight])



def on_press(key):
    global stop_training
    try:
        if key == keyboard.Key.esc:
            stop_training = True
            print("\nTraining interrupted by user.")
            return False
    except Exception as e:
        print(f"Error: {e}")


def train_model(articles_NN, threats_NN, combined_NN, dl_train, dl_val, device, num_epochs, lr, load_run=None):
    global stop_training

    # Load models from a specific run if specified
    if load_run is not None:
        run_folder = f'C:/Users/alon.zuaretz/Documents/GitHub/rockets-threat-prediction/results/run_{load_run}'
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

    pos_weight = calc_pos_weight(dl_train)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(list(articles_NN.parameters()) + list(threats_NN.parameters()) + list(combined_NN.parameters()), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    avg_loss_list = []
    avg_diff_loss_list = []
    val_loss_list = []
    val_avg_diff_list = []
    val_avg_class_list = []
    val_true_positive_list = []
    best_loss = float('inf')
    best_model = None

    # Create results directory if it doesn't exist
    if not os.path.exists(f'C:/Users/alon.zuaretz/Documents/GitHub/rockets-threat-prediction/results'):
        os.makedirs(f'C:/Users/alon.zuaretz/Documents/GitHub/rockets-threat-prediction/results')

    # Find the next available folder number
    folder_num = 1
    while os.path.exists(f'C:/Users/alon.zuaretz/Documents/GitHub/rockets-threat-prediction/results/run_{folder_num}'):
        folder_num += 1

    result_folder = f'C:/Users/alon.zuaretz/Documents/GitHub/rockets-threat-prediction/results/run_{folder_num}'
    os.makedirs(result_folder)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    num_of_ones = torch.zeros(1)
    num_of_zeros = torch.zeros(1)
    for articles_seq, threats_seq, labels in dl_val:
        num_of_ones += labels.sum()
        num_of_zeros += (labels.shape[0] * labels.shape[1] - labels.sum())

    for epoch in range(num_epochs):
        if stop_training:
            break

        start_time = time.time()
        epoch_loss = 0.0
        epoch_diff_loss = 0.0
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

            # Calculate average difference loss
            diff_loss = torch.mean(torch.abs(normalized_outputs - labels))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_diff_loss += diff_loss.item()
            num_batches += 1

            train_outputs.extend(normalized_outputs.detach().cpu().numpy())
            train_labels.extend(labels.detach().cpu().numpy())

        avg_loss = epoch_loss / num_batches
        avg_diff_loss = epoch_diff_loss / num_batches
        avg_loss_list.append(avg_loss)
        avg_diff_loss_list.append(avg_diff_loss)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model = {
                'articles_NN': articles_NN.state_dict(),
                'threats_NN': threats_NN.state_dict(),
                'combined_NN': combined_NN.state_dict()
            }

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}, Avg Difference Loss: {avg_diff_loss:.4f}, Time: {epoch_duration:.2f} seconds")

        # Validation every epoch
        if (epoch + 1) % 1 == 0:
            articles_NN.eval()
            threats_NN.eval()
            combined_NN.eval()
            val_loss = 0.0
            val_diff_loss = 0.0
            val_class_correct = 0
            val_total = 0
            val_true_positive = 0
            val_true_negative = 0
            val_batches = 0

            val_outputs = []
            val_labels = []

            with torch.no_grad():
                for articles_seq, threats_seq, labels in dl_val:
                    threats_seq = threats_seq.to(device)
                    articles_seq = articles_seq.to(device)
                    labels = labels.to(device)

                    articles_output = articles_NN(articles_seq)
                    threats_output = threats_NN(threats_seq)

                    outputs, normalized_outputs = combined_NN(articles_output, threats_output)

                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    # Calculate average difference loss (MAE)
                    diff_loss = torch.mean(torch.abs(normalized_outputs - labels))
                    val_diff_loss += diff_loss.item()

                    # Classify outputs around 0.5
                    predicted_classes = (normalized_outputs >= 0.5).float()
                    val_class_correct += (predicted_classes == labels).sum().item()
                    val_total += labels.numel()

                    # Calculate true positives (how many of the ones in the labels were correctly predicted)
                    val_true_positive += ((predicted_classes == 1) & (labels == 1)).sum().item()
                    val_true_negative += ((predicted_classes == 0) & (labels == 0)).sum().item()
                    val_batches += 1

                    val_outputs.extend(normalized_outputs.detach().cpu().numpy())
                    val_labels.extend(labels.detach().cpu().numpy())

            avg_val_loss = val_loss / val_batches
            avg_val_diff = val_diff_loss / val_batches
            val_accuracy = val_class_correct / val_total

            val_loss_list.append(avg_val_loss)
            val_avg_diff_list.append(avg_val_diff)
            val_avg_class_list.append(val_accuracy)
            val_true_positive_list.append(val_true_positive)

            print(f"Validation Loss after Epoch [{epoch + 1}]: {avg_val_loss:.4f}, Avg Difference Loss: {avg_val_diff:.4f}, \n"
                  f" Classification Accuracy: {val_accuracy:.4f}, True Positives out of all positives: {val_true_positive} / {num_of_ones} \n"
                  f" True Negatives out of all negatives: {val_true_negative} / {num_of_zeros} \n\n")

            articles_NN.train()
            threats_NN.train()
            combined_NN.train()

        # Step the scheduler
        scheduler.step()

    # Save the best model
    if best_model is not None:
        torch.save(best_model, os.path.join(result_folder, 'best_model.pth'))
        print(f"Best model saved as '{os.path.join(result_folder, 'best_model.pth')}'")

    # Save losses to .mat file
    scipy.io.savemat(os.path.join(result_folder, 'loss_data.mat'), {
        'avg_loss': avg_loss_list,
        'avg_diff_loss': avg_diff_loss_list,
        'val_loss': val_loss_list,
        'val_avg_diff': val_avg_diff_list,
        'val_avg_class': val_avg_class_list,
        'val_true_positive': val_true_positive_list
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
        'val_outputs': val_outputs,
        'val_labels': val_labels
    })
    print(f"Validation outputs vs labels saved to '{os.path.join(result_folder, f'outputs_vs_labels_val.mat')}'")

    print("Training complete.")
    return
