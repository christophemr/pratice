# Sentiment Analysis with IMDB Dataset

## Overview
This project is a sentiment analysis model trained on the IMDB dataset using TensorFlow and Keras. It utilizes an LSTM-based neural network to classify movie reviews as positive or negative.

## Features
- Uses IMDB dataset with 10,000 most common words
- LSTM layers for sequence processing
- Dropout layers to prevent overfitting
- Batch normalization for stability
- Adam optimizer with learning rate scheduling
- Early stopping to prevent overfitting

## Installation
### Prerequisites
Ensure you have Python installed along with the required libraries:

```sh
pip install tensorflow
```

## Usage
### Running the Script
Simply execute the script to train and evaluate the model:

```sh
python script.py
```

### Expected Output
The script will train the model and output validation accuracy and loss. It will also print the test accuracy upon completion.

## Model Architecture
- **Embedding Layer**: Converts word indices to dense vectors
- **LSTM Layer 1**: Processes sequences with 64 units
- **Dropout Layer**: Helps reduce overfitting
- **LSTM Layer 2**: Another LSTM with 32 units
- **Dropout Layer**: Further regularization
- **Dense Layer**: Fully connected layer with ReLU activation
- **Batch Normalization**: Stabilizes learning
- **Dropout Layer**: Additional regularization
- **Output Layer**: Single neuron with sigmoid activation

## Training Configuration
- **Optimizer**: Adam (learning rate 0.001)
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy
- **Batch Size**: 64
- **Epochs**: 10
- **Validation Split**: 20%
- **Callbacks**: Early Stopping, ReduceLROnPlateau

## License
This project is licensed under the MIT License.
