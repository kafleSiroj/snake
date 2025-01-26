import torch.nn as nn
import torch.nn.functional as F

class DQNNetwork(nn.Module):
    def __init__(self, input_lay, hidden_lay, output_lay):
        super().__init__()
        self.linear1 = nn.Linear(input_lay, hidden_lay)
        self.linear2 = nn.Linear(hidden_lay, output_lay)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x