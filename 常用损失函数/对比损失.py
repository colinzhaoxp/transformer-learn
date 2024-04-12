from cProfile import label
import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self, t):
        super().__init__()
        self.t =  1.0 / t
        self.ce = nn.CrossEntropyLoss()

    def forward(self, X):
        # X: batch * dim
        batch_size = X.shape[0]
        labels = torch.arange(start=0, end=batch_size, dtype=torch.long)
        labels = labels.to(X.device)

        images_sim = X @ X.t() * self.t

        loss = self.ce(images_sim, labels)

        return loss
