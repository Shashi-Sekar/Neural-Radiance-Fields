import torch

import numpy as np

class PositionEncoding:
    '''
    Positional Encoding for Neural Radiance Fields
    Transforms each component of the input coordinates to a higher dimensional space
    MLP needs to capture the high frequency variations in the scene (For handling high resolution images)
    Non-learnable function

    3D Spatial Coordinates
        Each x, y, z are transformed to 2L dimensional space
        L = 10
        Total: 60 dimensions
    Viewing Direction in terms of Unit Vector
        3 components are transformed to L dimensional space
        L = 4
        Total: 12 dimensions

    Parameters:
        L: Number of frequency bands
        x: Input coordinates (Batch_size, points, 3)
    Ouputs:
        x: Transformed coordinates (Batch_size, points, 6L)
    '''
    def __init__(self, L: int, include_inputs: bool) -> None:
        self.L = L
        self.inclue_inputs = include_inputs
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        freq_bands = 2.0 ** torch.linspace(0.0, self.L - 1, steps=self.L)
        
        sin_component = torch.sin(freq_bands * np.pi * x)
        cos_component = torch.cos(freq_bands * np.pi * x)
        
        embeddings = torch.stack([sin_component, cos_component], dim=-1)
        embeddings = embeddings.flatten()

        if self.include_inputs:
            embeddings = torch.cat([x, embeddings], dim=-1)

        return embeddings
    
