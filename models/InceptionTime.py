import torch
import torch.nn as nn

class InceptionModule(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_sizes=[9, 19, 39], bottleneck_channels=32):
        super(InceptionModule, self).__init__()
        
        self.use_bottleneck = in_channels > 1
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            b_channels = bottleneck_channels
        else:
            b_channels = in_channels

        self.convs = nn.ModuleList([
            nn.Conv1d(b_channels, out_channels, kernel_size=k, padding='same', bias=False) 
            for k in kernel_sizes
        ])

        self.maxconv = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        )

        self.bn = nn.BatchNorm1d(out_channels * 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        z = self.bottleneck(x) if self.use_bottleneck else x
        
        out = [conv(z) for conv in self.convs]
        
        out.append(self.maxconv(x))
        
        out = torch.cat(out, dim=1)
        
        return self.relu(self.bn(out))


class InceptionTime_FSL(nn.Module):

    def __init__(self, in_channels=1, num_blocks=6, out_channels=32, embedding_dim=128):
        super(InceptionTime_FSL, self).__init__()
        self.num_blocks = num_blocks
        
        self.blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        
        current_channels = in_channels
        for i in range(num_blocks):
            self.blocks.append(
                InceptionModule(
                    in_channels=current_channels, 
                    out_channels=out_channels,
                    kernel_sizes=[9, 19, 39]
                )
            )
            
            block_out_channels = out_channels * 4
            
            if i % 3 == 2:
                shortcut = nn.Sequential(
                    nn.Conv1d(in_channels, block_out_channels, kernel_size=1, padding='same', bias=False),
                    nn.BatchNorm1d(block_out_channels)
                )
                self.shortcuts.append(shortcut)
                in_channels = block_out_channels 
            
            current_channels = block_out_channels

        self.gap = nn.AdaptiveAvgPool1d(1)
        
        self.latent_projection = nn.Sequential(
            nn.Linear(current_channels, embedding_dim),
            nn.LayerNorm(embedding_dim) 
        )

    def forward(self, x):
        res_input = x
        shortcut_idx = 0
        
        for i in range(self.num_blocks):
            x = self.blocks[i](x)
            
            if i % 3 == 2:
                shortcut = self.shortcuts[shortcut_idx](res_input)
                x = torch.relu(x + shortcut)
                res_input = x
                shortcut_idx += 1
                
        x = self.gap(x)
        
        x = x.squeeze(-1)
        
        embeddings = self.latent_projection(x)
        
        return embeddings