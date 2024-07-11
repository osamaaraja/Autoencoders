# Autoencoder Models for Time Series Data Compression

### Description

This repository contains various autoencoder model architectures designed for compressing time series data from higher to lower dimensions. All models are implemented in PyTorch and leverage different neural network architectures to capture temporal and spatial features effectively.

### Model Architectures

1. **Fully Connected Layers:**
   - Autoencoders based on fully connected (dense) layers for direct compression of time series data.

2. **1D Convolutional Layers:**
   - Autoencoders utilizing 1D convolutional layers along the temporal dimension to capture temporal dependencies.

3. **2D Convolutional Layers:**
   - Autoencoders utilizing 2D convolutional layers to capture both temporal and spatial dimensions of the data.

4. **LSTM-based Autoencoders:**
   - Autoencoders that combine Long Short-Term Memory (LSTM) networks with 1D convolutional layers for feature extraction and temporal sequence modeling.

5. **Transformer-based Autoencoders:**
   - Autoencoders leveraging transformer architectures for efficient encoding and decoding of time series data, utilizing multihead-attention mechanisms.

### Features

- **Data Compression:** Efficiently reduce the dimensionality of time series data.
- **Temporal and Spatial Feature Extraction:** Capture complex dependencies and patterns in the data.
- **Versatility:** Apply to various types of time series data with different model architectures.
- **PyTorch Implementation:** All models are implemented using the PyTorch framework.


