import torch
import torch.nn as nn

class SqueezeExciteBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SqueezeExciteBlock, self).__init__()
        reduced_channels = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(reduced_channels, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class MLSTM_FCN_FSL(nn.Module):
    def __init__(self, in_channels, seq_len, lstm_hidden_dim=128, lstm_dropout_rate=0.8, embedding_dim=128, dimension_shuffle=False):
        super(MLSTM_FCN_FSL, self).__init__()
        self.dimension_shuffle = dimension_shuffle
        
        self.fcn_block1 = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=8, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            SqueezeExciteBlock(128)
        )
        
        self.fcn_block2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            SqueezeExciteBlock(256)
        )
        
        self.fcn_block3 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        if self.dimension_shuffle:
            lstm_input_size = seq_len
        else:
            lstm_input_size = in_channels
            
        self.lstm = nn.LSTM(
            input_size=lstm_input_size, 
            hidden_size=lstm_hidden_dim, 
            num_layers=1, 
            batch_first=True
        )
        self.lstm_dropout = nn.Dropout(lstm_dropout_rate)
        

        total_concat_dim = 128 + lstm_hidden_dim
        
        self.latent_projection = nn.Sequential(
            nn.Linear(total_concat_dim, embedding_dim),
            nn.LayerNorm(embedding_dim) 
        )

    def forward(self, x):
        
        out_fcn = self.fcn_block1(x)
        out_fcn = self.fcn_block2(out_fcn)
        out_fcn = self.fcn_block3(out_fcn)
        out_fcn = self.global_avg_pool(out_fcn).squeeze(-1) # Shape: [Batch, 128]
        
        if self.dimension_shuffle:
            lstm_in = x
        else:
            lstm_in = x.transpose(1, 2)
            
        lstm_out, (h_n, c_n) = self.lstm(lstm_in)
        out_lstm = self.lstm_dropout(h_n[-1]) 
        
        concat_out = torch.cat([out_fcn, out_lstm], dim=1)
        embeddings = self.latent_projection(concat_out)
        
        return embeddings