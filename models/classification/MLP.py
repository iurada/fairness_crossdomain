import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, dropout=0.5, width=256, depth=3):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, width)
        self.dropout = nn.Dropout(dropout)
        self.hiddens = nn.ModuleList([
            nn.Linear(width, width)
            for _ in range(depth-2)])
        self.output = nn.Linear(width, n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x