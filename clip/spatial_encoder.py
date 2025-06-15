import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class SE_MLP(nn.Module):
    def __init__(self, original_coord_dim=2, output_dim=768):
        super(SE_MLP, self).__init__()
        self.fc1 = nn.Linear(original_coord_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SE_Sinusoidal_MLP(nn.Module):
    def __init__(self, original_coord_dim=2, sinusoidal_dim=10, output_dim=768):
        super(SE_Sinusoidal_MLP, self).__init__()
        
        self.original_coord_dim = original_coord_dim # 元の座標次元 (例: 2 for (x,y))
        self.sinusoidal_dim = sinusoidal_dim         # sinusoidal_feature の dim
        
        sinusoidal_output_size = self.original_coord_dim * self.sinusoidal_dim * 2
        
        # MLP層
        self.fc = nn.Linear(sinusoidal_output_size, output_dim)

    def sinusoidal_feature(self, x_coords):
        x_coords = x_coords.squeeze(1)
        x_coords_normalized = torch.stack([
            (x_coords[:, 0] - 2) / np.sqrt(2),
            (x_coords[:, 1] - 1.5) / np.sqrt(1.25)
        ], dim=1)

        freq_bands = 2**torch.arange(self.sinusoidal_dim, device=x_coords.device, dtype=x_coords.dtype) * np.pi

        features_pre_sin_cos = x_coords_normalized.unsqueeze(2) * freq_bands

        # sin と cos を計算
        sin_features = torch.sin(features_pre_sin_cos)
        cos_features = torch.cos(features_pre_sin_cos)

        feature = torch.cat([sin_features, cos_features], dim=2).flatten(start_dim=1)
        
        return feature
            
    def forward(self, x):
        x = self.sinusoidal_feature(x)
        x = self.fc(x)
        return x