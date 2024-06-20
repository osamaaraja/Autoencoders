"This script contains the definition of the model"


import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.heads * self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.heads * self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.heads * self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries, mask):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.nn.functional.softmax(energy, dim=3)

        out = torch.einsum("nhql,nlhd->nhqd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, 2 * embed_size),
            nn.ReLU(),
            nn.Linear(2 * embed_size, embed_size),
        )

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.norm1(attention + query)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return out


class Autoencoder(nn.Module):
    def __init__(self, heads, input_dim, num_layers, latent_dim, sequence_length, use_batch_norm=False):
        super(Autoencoder, self).__init__()

        self.latent_dim = latent_dim
        self.use_batch_norm = use_batch_norm

        # Convolutional layers for encoder
        self.conv1 = nn.Conv1d(input_dim, 12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(12) if use_batch_norm else nn.Identity()
        self.conv2 = nn.Conv1d(12, 8, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(8) if use_batch_norm else nn.Identity()
        self.conv3 = nn.Conv1d(8, 6, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(6) if use_batch_norm else nn.Identity()

        self.to_latent = nn.Linear(6, latent_dim)

        self.from_latent = nn.Linear(latent_dim, 6)

        # Decoder linear layers
        self.conv4 = nn.ConvTranspose1d(6, 8, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(8) if use_batch_norm else nn.Identity()
        self.conv5 = nn.ConvTranspose1d(8, 12, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(12) if use_batch_norm else nn.Identity()
        self.conv6 = nn.ConvTranspose1d(12, input_dim, kernel_size=3, stride=1, padding=1)

        self.encoders = nn.ModuleList([TransformerBlock(6, heads) for _ in range(num_layers)])
        self.decoders = nn.ModuleList([TransformerBlock(6, heads) for _ in range(num_layers)])

    def forward(self, x, mask=None):

        # Transpose input to (batch_size, channels, sequence_length)
        x = x.transpose(1, 2)

        # Convolutional encoder
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.permute(0,2,1)

        for encoder in self.encoders:
            x = encoder(x, x, x, mask)

        # Reshape for the linear layer
        batch_size, seq_len, features = x.shape
        x = x.view(batch_size * seq_len, features)

        # Apply linear layer to reduce to latent dimension
        latent_representation = torch.relu(self.to_latent(x))

        x = torch.relu(self.from_latent(latent_representation))

        # Reshape back to sequence format
        x = x.view(batch_size, seq_len, -1)

        for decoder in self.decoders:
            x = decoder(x, x, x, mask)

        x = x.permute(0, 2, 1)

        # Transposed convolutional decoder
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.conv6(x))

        x = x.transpose(1, 2)

        latent_representation = latent_representation.view(batch_size, seq_len, -1)

        return x, latent_representation




