# DL for NLP - Transformer

This repository contains the implementation of **Exercise 2** for the "Deep Learning for NLP" course. The exercise focuses on building, tuning, and evaluating a Transformer model from scratch, inspired by [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762). The project is structured as per the initial template provided to students, clearly separating instructor-provided code and personal contributions.

## Overview

The project includes:
- Implementation of the original Transformer architecture, including:
  - Encoder and Decoder modules.
  - Multi-Head Self-Attention mechanism.
  - Positional Encoding.
- Tokenization and preprocessing of text data.
- Custom loss functions and training loops.
- Evaluation of model performance using various metrics.
- Hyperparameter tuning for optimization.

This repository adheres to the course submission guidelines and retains the original structure as required for evaluation.

## My Contributions

While the initial template was provided as a foundation, my contributions include:
- Development of core Transformer components (e.g., multi-head attention, positional encoding, etc.).
- Implementation of tokenization and vocabulary building.
- Creating custom training and validation loops using PyTorch.
- Optimizing hyperparameters and experimenting with learning rates, batch sizes, and dropout rates.
- Performing model evaluation and presenting results through detailed analysis.

## Key Features

### Transformer Architecture
- **Encoder-Decoder**: Implemented from scratch, including self-attention and feed-forward submodules.
- **Positional Encoding**: Added positional information to input embeddings.
- **Multi-Head Attention**: Enabled the model to focus on different parts of input sequences simultaneously.

### Dataset Preparation
- Tokenized and preprocessed the dataset into sequences compatible with the Transformer.
- Built a vocabulary with a fixed size using token frequency thresholds.

### Training and Evaluation
- Developed custom training and validation loops.
- Used early stopping to prevent overfitting.
- Measured performance with loss and accuracy metrics.

## Requirements

The code was tested on the following dependencies:
- `torch==2.0.1`
- `numpy==1.23.5`
- `matplotlib==3.7.1`
- `seaborn==0.12.2`

## Notes

- The `.ipynb` notebook file contains the results of the training, validation, and evaluation processes.
- The `.py` file is an exported version of the notebook, formatted for submission and compliance with course guidelines.
- The project retains the structure provided to align with course requirements.

## How to Run

1. Install the required dependencies listed above.
2. Run the Python file or open the notebook to explore the code.
3. Follow the instructions in the notebook to train and evaluate the model.

## References

- Vaswani, A., et al. (2017). Attention is All You Need. [Link to Paper](https://arxiv.org/abs/1706.03762).
- Course materials and initial template provided by the instructors.
