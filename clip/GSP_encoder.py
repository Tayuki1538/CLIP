import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class GE_1DCNN(nn.Module):
    def __init__(self, input_length=4800, output_dim=512):
        super(GE_1DCNN, self).__init__()

        # 入力チャンネル数は1
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        
        # Block 1: 比較的大きなカーネルで初期の特徴抽出
        self.conv_block1 = nn.Sequential(
            self.conv1, # 4800 -> 2400
            nn.BatchNorm1d(32),
            nn.ReLU(),
            self.conv2, # 2400 -> 2400
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4) # 2400 -> 600
        )
        
        # Block 2: チャンネル数拡大とダウンサンプリング
        self.conv_block2 = nn.Sequential(
            self.conv3, # 600 -> 300
            nn.BatchNorm1d(64),
            nn.ReLU(),
            self.conv4, # 300 -> 300
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4) # 300 -> 75
        )

        # Block 3: 更なるチャンネル拡大とダウンサンプリング
        self.conv_block3 = nn.Sequential(
            self.conv5, # 75 -> 38 (端数処理に注意)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            self.conv6, # 38 -> 38
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2) # 38 -> 19
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1) # [batch, 128, 19] -> [batch, 128, 1]
        
        # 全結合層
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
