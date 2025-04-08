# Deep Attribution Modeling with LSTM + Attention (Simplified Example)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Sample touchpoint vocab and mapping
touchpoint_vocab = ['google_ad', 'email', 'app_push', 'site_visit', 'facebook_ad']
vocab_size = len(touchpoint_vocab)
tp_to_idx = {tp: i for i, tp in enumerate(touchpoint_vocab)}

# Hyperparameters
embedding_dim = 16
hidden_dim = 32
max_len = 5
batch_size = 2
epochs = 10
learning_rate = 0.001

# Dummy dataset class
class TouchpointDataset(Dataset):
    def __init__(self, data):
        self.sequences = []
        self.labels = []
        for seq, label in data:
            encoded = [tp_to_idx.get(tp, 0) for tp in seq]
            padded = encoded + [0] * (max_len - len(encoded))
            self.sequences.append(padded[:max_len])
            self.labels.append(label)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.labels[idx], dtype=torch.float32)

# Attention Layer
def attention_layer(hidden_states):
    # hidden_states: (batch_size, seq_len, hidden_dim)
    scores = torch.tanh(hidden_states)  # (B, T, H)
    weights = torch.softmax(scores @ torch.randn(hidden_dim, 1), dim=1)  # (B, T, 1)
    context = torch.sum(hidden_states * weights, dim=1)  # (B, H)
    return context, weights.squeeze(-1)

# LSTM + Attention model
class AttributionLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x):  # x: (B, T)
        emb = self.embedding(x)  # (B, T, D)
        h, _ = self.lstm(emb)  # (B, T, H)
        context, weights = attention_layer(h)  # (B, H), (B, T)
        out = torch.sigmoid(self.output(context))  # (B, 1)
        return out.squeeze(1), weights

# Sample training data
data = [
    (["google_ad", "email", "app_push"], 1),
    (["facebook_ad", "site_visit", "email"], 0),
    (["email", "app_push"], 1),
    (["site_visit", "google_ad"], 0)
]

dataset = TouchpointDataset(data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = AttributionLSTM(vocab_size, embedding_dim, hidden_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()
        pred, _ = model(x_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    for x_batch, y_batch in dataloader:
        pred, attn_weights = model(x_batch)
        print("Input:", x_batch)
        print("Predicted:", pred)
        print("Attention Weights:", attn_weights)
        print("True Labels:", y_batch)
