"This script contains the definition of the model"


import torch
import torch.nn as nn
import torch.nn.functional as F


class StackedSparseAutoEncoder(nn.Module):
    def __init__(self, input_dim, input_length, hidden_dims, dropout_rate, sparsity_penalty, target_sparsity,
                 batch_norm=False):
        super().__init__()

        self.sparsity_penalty = sparsity_penalty
        self.target_sparsity = target_sparsity
        self.input_length = input_length
        self.batch_norm = batch_norm

        # Define encoder layers
        self.encoder = nn.ModuleList()
        prev_dim = input_dim
        for dim in hidden_dims:
            self.encoder.append(nn.Conv1d(prev_dim, dim, kernel_size=3, stride=1, padding=1))
            if self.batch_norm:
                self.encoder.append(nn.BatchNorm1d(dim))  # Conditionally add BatchNorm layer
            self.encoder.append(nn.ReLU())
            self.encoder.append(nn.Dropout(dropout_rate))
            prev_dim = dim

        # Define decoder layers
        self.decoder = nn.ModuleList()
        for i in range(len(hidden_dims) - 1, 0, -1):
            self.decoder.append(
                nn.ConvTranspose1d(hidden_dims[i], hidden_dims[i - 1], kernel_size=3, stride=1, padding=1))
            if self.batch_norm:
                self.decoder.append(nn.BatchNorm1d(hidden_dims[i - 1]))  # Conditionally add BatchNorm layer
            self.decoder.append(nn.ReLU())
            self.decoder.append(nn.Dropout(dropout_rate))

        # Last layer to reconstruct original number of channels
        self.decoder.append(nn.ConvTranspose1d(hidden_dims[0], input_dim, kernel_size=3, stride=1, padding=1))
        self.decoder.append(nn.ReLU())

    def forward(self, x):
        # Encoding
        for layer in self.encoder:
            x = layer(x)
        encoded_features = torch.relu(x)

        # Decoding
        for layer in self.decoder:
            x = layer(x)

        return x, encoded_features


    def sparse_loss(self, reconstructed_x, x, encoded_features):
        # Compute reconstruction loss (MSE)
        mse_loss = F.mse_loss(reconstructed_x.float(), x.float(), reduction='sum')

        # Compute L1 penalty (sparsity constraint)
        avg_output = torch.mean(encoded_features, dim=0)  # Average over samples in the batch
        l1_penalty = torch.mean(torch.abs(avg_output - self.target_sparsity))

        # Combine MSE loss and L1 penalty
        total_loss = mse_loss + self.sparsity_penalty * l1_penalty

        return total_loss



