import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class GD_MLP(nn.Module):
    def __init__(self, input_dim=512, output_dim=2):
        super(GD_MLP, self).__init__()

        # 入力チャンネル数は1
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        return x
    
