import torch

import torch.nn as nn
import torch.nn.functional as F

class NeRF(nn.Module):
    def __init__(self, input_dim=60, hidden_ch=256, second_hidden_ch=128, out_ch=3, view_dirs_dim=24):
        super().__init__()

        self.input_dim = input_dim
        self.view_dirs_dim = view_dirs_dim

        self.density_mlp_first = nn.ModuleList([
            nn
        ])
        self.density_mlp_first = nn.Sequential(
            nn.Linear(input_dim, hidden_ch),
            nn.ReLU(),
            nn.Linear(hidden_ch, hidden_ch),
            nn.ReLU(),
            nn.Linear(hidden_ch, hidden_ch),
            nn.ReLU(),
            nn.Linear(hidden_ch, hidden_ch),
            nn.ReLU(),
            nn.Linear(hidden_ch, hidden_ch),
            nn.ReLU(),
        )

        self.density_mlp_second = nn.Sequential(
            nn.Linear(input_dim + hidden_ch, hidden_ch),
            nn.ReLU(),
            nn.Linear(hidden_ch, hidden_ch),
            nn.ReLU(),
            nn.Linear(hidden_ch, hidden_ch),
            nn.ReLU(),
        )

        self.volume_layer = nn.Linear(hidden_ch, 1)
        self.feature_layer = nn.Linear(hidden_ch, hidden_ch)
        self.color_mlp = nn.Sequential(
            nn.Linear(view_dirs_dim+hidden_ch, second_hidden_ch),
            nn.ReLU(),
            nn.Linear(second_hidden_ch, out_ch),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x, view_dirs = torch.split(x, [self.input_dim, self.view_dirs_dim], dim=-1)

        #Multi-view Consistent Density
        out = self.density_mlp_first(x)
        out = self.density_mlp_second(torch.cat([x+out], dim=-1))

        #Volume Density
        density = self.volume_layer(out)
        density = F.relu(density)   

        #Feature
        features = self.feature_layer(out)

        #Color
        color = self.color_mlp(torch.cat(view_dirs, features), dim=-1)
        
        return torch.cat([color, density], dim=-1)