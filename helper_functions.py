'''
This script contains the utilities for the 5 different scripts
'''

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import LabelEncoder
import json
import csv
import logging
import copy

from .model import Autoencoder
from .hyper_params import *

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_out', nonlinearity='relu')

def save_best_params(best_params, filename):
    with open(filename, 'w') as f:
        json.dump(best_params, f)
    logging.info(f"Best parameters saved to {filename}")

def calculate_r2(original, reconstructed):
    logging.info("Calculating R² score.")
    # Calculate the mean of the original signals
    mean_original = np.mean(original, axis=0)

    # Calculate SSE (sum of squared errors)
    SSE = np.sum((original - reconstructed) ** 2)

    # Calculate SST (sum of squared total)
    SST = np.sum((original - mean_original) ** 2)

    # Calculate R² score
    R2 = 1 - (SSE / SST)

    return R2

def grid_search(param_grid, train_loader, val_loader):
    logging.info("Performing Grid search.")
    best_performance = float('inf')
    best_params = None

    for params in ParameterGrid(param_grid):
        logging.info(f"Training with parameters: {params}")

        model = Autoencoder(
            heads=heads,
            input_dim=input_dim,
            num_layers=n_layer,
            latent_dim=latent_dim,
            sequence_length=window_size,
            use_batch_norm=True

        ).to(device)
        model.apply(init_weights)

        # Train the model with new learning rate
        train_losses, val_losses = train_subscript(model, train_loader, val_loader, num_epochs_for_grid_search, lr=params['lr'])

        current_performance = val_losses[-1]
        logging.info(f"Validation loss for parameters {params}: {current_performance}")

        if current_performance < best_performance:
            best_performance = current_performance
            best_params = params
    logging.info(f"Best parameters found: {best_params}")
    return best_params

def save_losses(train_losses, val_losses, filename, training_participants=None, validating_participant=None,
                include_participants=False):
    logging.info("Saving the training and validation losses.")
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Conditionally add headers based on the include_participants flag
        if include_participants and training_participants is not None and validating_participant is not None:
            writer.writerow(
                ['Epoch', 'Train Loss', 'Validation Loss', 'Training Participants', 'Validating Participant'])
        else:
            writer.writerow(['Epoch', 'Train Loss', 'Validation Loss'])

        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), start=1):
            # Conditionally add participant information
            if include_participants and training_participants is not None and validating_participant is not None:
                writer.writerow([epoch, train_loss, val_loss, str(training_participants), validating_participant])
            else:
                writer.writerow([epoch, train_loss, val_loss])

def load_data(file_path_or_participant, sequence_length, shuffle, split=False, include_labels=True):
    logging.info("Loading and windowing data.")

    # Determine whether a file path or participant ID is provided
    if isinstance(file_path_or_participant, (float, int)):  # Participant ID provided
        file_path = f'{EMG_data_path}/new_participant_{file_path_or_participant}_data_1Hz.csv'

    else:  # File path provided
        file_path = file_path_or_participant

    data = pd.read_csv(file_path, low_memory=False)

    label_list = data['label'].unique()
    encoded_labels = label_encoder.fit_transform(data['label']) if 'label' in data.columns else None
    numeric_data = np.array(data.filter(like="EMG"))

    # Reshape data based on sequence length using sliding window approach
    sequences = []
    labels = []

    step_size = 1  # Step size for overlapping windows

    for i in range(0, len(numeric_data) - sequence_length + 1, step_size):
        sequences.append(numeric_data[i:i + sequence_length])
        labels.append(encoded_labels[i:i + sequence_length])

    if split:
        train_data, val_data = train_test_split(sequences, test_size=0.4, shuffle=shuffle, random_state=42)
        if include_labels:
            train_labels, val_labels = train_test_split(labels, test_size=0.4, shuffle=shuffle, random_state=42)
        train_tensor = torch.tensor(train_data, dtype=torch.float32).to(device)
        val_tensor = torch.tensor(val_data, dtype=torch.float32).to(device)
        train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(TensorDataset(val_tensor), batch_size=batch_size)
        return (train_loader, val_loader, train_data, train_labels, label_list) if include_labels else (train_loader, val_loader)

    else:
        data_tensor = torch.tensor(sequences, dtype=torch.float32).to(device)
        loader = DataLoader(TensorDataset(data_tensor), batch_size=batch_size, shuffle=shuffle)
        return loader, sequences, labels, label_list if include_labels else loader

def train_subscript(model, train_loader, val_loader, num_epochs, lr):
    logging.info("Training begins.")
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20,
                                                     factor=0.5)
    # empty lists for gathering losses
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 20
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            inputs = batch[0].to(device)
            outputs, encoded = model(inputs)
            loss = criterion(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        val_loss = validate_model(model, val_loader)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                logging.info("Early stopping triggered.")
                model.load_state_dict(best_model_state)
                break

    return train_losses, val_losses

def train_model(model, train_loader, val_loader, num_epochs, lr,  training_participants, validating_participant, folder_path_for_values):

    train_losses, val_losses=train_subscript(model, train_loader, val_loader, num_epochs, lr)

    save_losses(train_losses, val_losses,
                f'{folder_path_for_values}/loss_data_based_on_BestParams_trainingPartcipants_{training_participants}_ValParticipants_{validating_participant}.csv',
                training_participants, validating_participant, True)

def AE_train_with_split(model, train_loader_fine_tuning, val_loader_fine_tuning,num_epochs,lr, train_data_fine_tuning, folder_path_for_values, pretrained):
    train_losses, val_losses = train_subscript(model, train_loader_fine_tuning, val_loader_fine_tuning, num_epochs, lr)

    if pretrained:

        save_losses(train_losses, val_losses, f'{folder_path_for_values}/loss_data_for_fine_tuning_participant_with_pretrained_weights.csv')

    else:

        save_losses(train_losses, val_losses, f'{folder_path_for_values}/loss_data_for_fine_tuning_participant_with_init_weights.csv')


    # Evaluate the model without computing gradients
    model.eval()
    encoded_features_list = []
    decoded_data_list = []

    with torch.no_grad():
        for batch_data, in train_loader_fine_tuning:  # Added comma to unpack the batch data
            # Move the input data to device
            inputs = batch_data.to(device)

            # Forward pass through the model
            decoded_data, encoded_features = model(inputs)

            # Collect the results
            encoded_features_list.append(encoded_features.cpu().numpy())
            decoded_data_list.append(decoded_data.cpu().numpy())

    # Concatenate the results from all batches
    encoded_features_all = np.vstack(encoded_features_list)
    decoded_data_all = np.vstack(decoded_data_list)

    # Calculate and save R2 score
    r2_ae = calculate_r2(train_data_fine_tuning, decoded_data_all)

    return encoded_features_all, decoded_data_all, r2_ae

def validate_model(model, val_loader):
    logging.info("Evaluating the model.")
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            val_inputs = batch[0].to(device)
            val_outputs, val_enc = model(val_inputs)
            val_loss += criterion(val_outputs, val_inputs).item()
    val_loss /= len(val_loader)
    return val_loss





