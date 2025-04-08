Deep Attribution Modeling with LSTM + Attention

This repository demonstrates a simple yet effective approach to user journey attribution modeling using LSTM + Attention in PyTorch.

 Overview

Attribution modeling aims to assign credit to each marketing channel (touchpoint) in a user's path that leads to a conversion. This project uses a deep learning model with attention to learn which steps in the sequence contributed most to conversion.

 Model Architecture

Embedding Layer: transforms touchpoints into dense vectors

LSTM Layer: processes sequences of touchpoints

Attention Layer: learns importance weights of each touchpoint

Output Layer: sigmoid layer for binary classification (conversion or not)

 Features

Custom TouchpointDataset class

Padding and truncation for fixed sequence length

Attention mechanism for interpretability

Outputs attention weights per touchpoint

Sample Input/Output

Input: ['google_ad', 'email', 'app_push']
Prediction: 0.83
Attention: [0.52, 0.31, 0.17]
