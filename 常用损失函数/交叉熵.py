import torch
import torch.nn as nn

"""
从零实现一个简单的交叉熵损失函数
"""

class CrossEntropyLoss(nn.Module):
    def __init__(self, t):
        super().__init__()
        self.t = 1.0 / t
 
    def forward(self, X, y):
        # X: batch * n_class
        # y: batch
        exp_preds = torch.exp(X)
        logits = exp_preds / exp_preds.sum(dim=1, keepdim=True)
        logits = logits * self.t

        batch_size = X.shape[0]
        loss = -torch.log(logits[range(batch_size), y])
        loss = torch.sum(loss) / batch_size

        return loss
