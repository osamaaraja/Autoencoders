"This script contains the definition of the model"


import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, enc_hidden_dims, dec_hidden_dims, latent_dim, use_batch_norm=False):
        super(LSTMAutoencoder, self).__init__()

        self.use_batch_norm = use_batch_norm

        # Convolutional layers for compression in encoder
        self.conv1 = nn.Conv1d(input_size, enc_hidden_dims[0], kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(enc_hidden_dims[0]) if use_batch_norm else nn.Identity()
        self.conv2 = nn.Conv1d(enc_hidden_dims[0], enc_hidden_dims[1], kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(enc_hidden_dims[1]) if use_batch_norm else nn.Identity()

        # Encoder LSTM block
        self.encoder_lstm = nn.LSTM(enc_hidden_dims[1], enc_hidden_dims[2], batch_first=True)

        # Transition from encoder to latent space
        self.fc_encoder_to_latent = nn.Linear(enc_hidden_dims[2], latent_dim)

        # Transition from latent space to decoder
        self.fc_latent_to_decoder = nn.Linear(latent_dim, dec_hidden_dims[0])

        # Decoder LSTM block
        self.decoder_lstm = nn.LSTM(dec_hidden_dims[0], dec_hidden_dims[1], batch_first=True)

        # Transposed convolutional layers for expansion in decoder
        self.conv_transpose1 = nn.ConvTranspose1d(dec_hidden_dims[1], dec_hidden_dims[2], kernel_size=3, stride=1, padding=1)
        self.bn_transpose1 = nn.BatchNorm1d(dec_hidden_dims[2]) if use_batch_norm else nn.Identity()
        self.conv_transpose2 = nn.ConvTranspose1d(dec_hidden_dims[2], input_size, kernel_size=3, stride=1, padding=1)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # Convolutional encoder
        x = x.transpose(1, 2)  # Transpose to (batch_size, channels, sequence_length)

        x = F.relu(self.bn1(self.conv1(x)))

        x = F.relu(self.bn2(self.conv2(x)))

        x = x.transpose(1, 2)  # Transpose back to (batch_size, sequence_length, channels)

        # Encoder LSTM pass
        x, _ = self.encoder_lstm(x)

        x = self.dropout(x)

        # Reshape for the linear layer
        batch_size, seq_len, features = x.shape
        x = x.reshape(batch_size * seq_len, features)

        latent_representation = torch.relu(self.fc_encoder_to_latent(x))

        # Reshape back to sequence format for decoder
        x = torch.relu(self.fc_latent_to_decoder(latent_representation)).reshape(batch_size, seq_len, -1)

        # Decoder LSTM pass
        x, _ = self.decoder_lstm(x)

        x = self.dropout(x)

        # Reshape for transposed convolutional layers
        x = x.transpose(1, 2)  # Transpose for convolutional layers

        x = F.relu(self.bn_transpose1(self.conv_transpose1(x)))

        x = F.relu(self.conv_transpose2(x))
        x = x.transpose(1, 2)  # Transpose back to original

        latent_representation = latent_representation.view(batch_size, seq_len, -1)

        return x, latent_representation




