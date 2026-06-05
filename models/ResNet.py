import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=8, padding='same')
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding='same')
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        self.relu = nn.ReLU()
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same'),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        res = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += res
        out = self.relu(out)
        
        return out


class ResNet_FSL(nn.Module):

    def __init__(self, in_channels, embedding_dim=128):
        super(ResNet_FSL, self).__init__()
        

        self.block1 = ResNetBlock(in_channels, 64)
        
        self.block2 = ResNetBlock(64, 128)
        
        self.block3 = ResNetBlock(128, 128)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.latent_projection = nn.Sequential(
            nn.Linear(128, embedding_dim),
            nn.LayerNorm(embedding_dim) 
        )

    def forward(self, x):
        """
        x shape: [Batch_size, in_channels, Sequence_length]
        """
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        
        out = self.global_avg_pool(out).squeeze(-1) # Shape: [Batch, 128]
        
        embeddings = self.latent_projection(out)
        return embeddings