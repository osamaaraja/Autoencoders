"This script contains the definition of the model"


import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    def __init__(self, layer_sizes, latent_dim, l1_penalty, target_sparsity, dropout_rate):
        super(SparseAutoencoder, self).__init__()

        assert len(layer_sizes) > 1, "layer_sizes list must contain at least input and one hidden layer size"

        # Encoder
        encoder_layers = []
        for i in range(len(layer_sizes) - 1):
            encoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            encoder_layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))  # Add BatchNorm layer
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout_rate))
        # Add the latent layer
        encoder_layers.append(nn.Linear(layer_sizes[-1], latent_dim))
        encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = [nn.Linear(latent_dim, layer_sizes[-1]), nn.ReLU(), nn.Dropout(dropout_rate)]
        for i in range(len(layer_sizes) - 1, 0, -1):
            decoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i - 1]))
            if i > 1:  # Add BatchNorm, ReLU, and Dropout to all but the last layer
                decoder_layers.append(nn.BatchNorm1d(layer_sizes[i - 1]))  # Add BatchNorm layer
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Dropout(dropout_rate))
        self.decoder = nn.Sequential(*decoder_layers)

        self.l1_penalty = l1_penalty
        self.target_sparsity = target_sparsity
        self.sparsity_scale = 0.01

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

    def kl_divergence(self, rho, rho_hat):
        return torch.sum(rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat)))

    def loss_function(self, inputs, outputs, encoded):
        mse_loss = nn.functional.mse_loss(outputs, inputs)
        rho_hat = torch.mean(encoded, dim=0)
        sparsity_loss = self.sparsity_scale * self.kl_divergence(self.target_sparsity, rho_hat)
        l1_loss = sum(p.abs().sum() for p in self.parameters())
        total_loss = mse_loss + sparsity_loss + self.l1_penalty * l1_loss
        return total_loss




