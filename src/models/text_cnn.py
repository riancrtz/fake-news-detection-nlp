"""
text_cnn.py
Text-CNN following Kim (2014).
Convolutional neural networks for sentence classification.
"""

import torch
import torch.nn as nn


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes,
                 kernel_sizes=[3, 4, 5], num_filters=128, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = [torch.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        return self.fc(x)
