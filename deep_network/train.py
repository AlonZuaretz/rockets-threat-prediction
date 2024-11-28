import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.io
import csv
import os
import torch.nn.functional as F
import sys

from torch.optim.lr_scheduler import StepLR
from pynput import keyboard

stop_training = False


def on_press(key):
    global stop_training
    try:
        if key == keyboard.Key.esc:
            stop_training = True
            print("\nTraining interrupted by user.")
            return False
    except Exception as e:
        print(f"Error: {e}")


def train_model(articles_NN, threats_NN, combined_NN, dl, device, num_epochs, lr, load_run=None):
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

    criterion = nn.BCELoss(reduction='mean')
    optimizer = torch.optim.Adam(list(articles_NN.parameters()) + list(threats_NN.parameters()) + list(combined_NN.parameters()), lr=lr)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    avg_loss_list = []
    avg_diff_loss_list = []
    best_loss = float('inf')
    best_model = None

    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')

    # Find the next available folder number
    folder_num = 1
    while os.path.exists(f'results/run_{folder_num}'):
        folder_num += 1

    result_folder = f'results/run_{folder_num}'
    os.makedirs(result_folder)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    for epoch in range(num_epochs):
        if stop_training:
            break

        start_time = time.time()
        epoch_loss = 0.0
        epoch_diff_loss = 0.0
        num_batches = 0

        all_outputs = []
        all_labels = []

        for (articles_seq, threats_seq, labels) in dl:
            threats_seq = threats_seq.to(device)
            articles_seq = articles_seq.to(device)
            labels = labels.to(device)

            articles_output = articles_NN(articles_seq)
            threats_output = threats_NN(threats_seq)

            outputs = combined_NN(articles_output, threats_output)

            loss = criterion(outputs, labels)

            # Calculate average difference loss
            diff_loss = torch.mean(torch.abs(outputs - labels))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_diff_loss += diff_loss.item()
            num_batches += 1

            all_outputs.extend(outputs.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

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

        # Step the scheduler
        scheduler.step()

    # Save the best model
    if best_model is not None:
        torch.save(best_model, os.path.join(result_folder, 'best_model.pth'))
        print(f"Best model saved as '{os.path.join(result_folder, 'best_model.pth')}'")

    # Save losses to .mat file
    scipy.io.savemat(os.path.join(result_folder, 'loss_data.mat'), {'avg_loss': avg_loss_list, 'avg_diff_loss': avg_diff_loss_list})
    print(f"Loss data saved to '{os.path.join(result_folder, 'loss_data.mat')}'")

    # Plot the loss graphs
    plt.figure()
    plt.plot(range(1, len(avg_loss_list) + 1), avg_loss_list, label='Average BCE Loss')
    plt.plot(range(1, len(avg_diff_loss_list) + 1), avg_diff_loss_list, label='Average Difference Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss over Epochs')
    plt.savefig(os.path.join(result_folder, 'loss_plot.png'))
    plt.show()

    # Save outputs vs labels to CSV
    with open(os.path.join(result_folder, 'outputs_vs_labels.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Output', 'Label'])
        for output, label in zip(all_outputs, all_labels):
            writer.writerow([output, label])
    print(f"Outputs vs labels saved to '{os.path.join(result_folder, 'outputs_vs_labels.csv')}'")

    print("Training complete.")
    return
