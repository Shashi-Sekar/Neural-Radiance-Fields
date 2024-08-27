import torch
import numpy as np

def ray_generation(H: int, W: int, focal_length: float, c2w: torch.Tensor) -> torch.Tensor:
    '''
    Generate the ray origins and directions for each pixel in the image
    Parameter:
        H: Height of the image
        W: Width of the image
        focal_length: Focal length of the camera
        c2w: Camera to world transformation matrix (4, 4) -> [R | t][0,0,0,1]
    Outputs:
        ray_origins: Ray origins for each pixel in the image (H, W, 3)
        ray_directions: Ray directions for each pixel in the image (H, W, 3)
    '''

    #Get the image coordinates
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32), torch.arange(H, dtype=torch.float32), indexing='xy')

    #Center the image aroud the origin (center of the image plane)
    i = i - W/2
    j = j - H/2

    #Scaling the image coorindates
    i = i / focal_length
    j = j / focal_length

    #Compute the ray directions
    #(x, y, -1) - (0, 0, 0) -> (x, y, -1)
    #Camera orientation in the negative z-direction
    #(W, H, 3)
    ray_directions = torch.stack([i, j, -torch.ones_like(i)], dim=-1)
    
    #Tranform the ray directions from camera space to world space
    #(W, H, 1, 3) * (3, 3) (Broadcast) -> (W, H, 3, 3) ->(Sum) (W, H, 3)
    ray_directions_world = torch.sum(ray_directions[..., np.newaxis, :] * c2w[:3, :3], dim=-1)

    #Ray Origin -> Translation of the camera from the world origin
    #Last column of c2w matrix
    ray_origins = c2w[:3, -1].expand(ray_directions_world.shape)

    return ray_origins, ray_directions_world

def hierarchical_sampling(bins: int, weights: torch.tensor, num_samples: int, uniform=False):
    '''
    Hierarchical sampling -> to sample more points from those positions we expect to have visual content
    Based on the evaluation of the 'coarse' network on stratified samples
    Parameters:
        bins:           Number of bins used for stratified sampling
        weights:        Weights for the bins
        num_samples:    Number of samples
        uniform:        Whether to perform uniform sampling
    '''

    pdf = weights / torch.sum(weights, dim=-1)

    pass
def rays_ndc(H, W, focal, near, rays_origin, rays_direction):
    '''
    Generate the rays in NDC space (Normalized Device Coordinates)
    This is commonly used in triangle rasterization pipeline
    Clipping out points that lie before the near plane
    Parameters:
        H: Height of the image
        W: Width of the image
        focal: Focal length of the camera
        near: Near dpeth bound for the scene
        rays_origin: Ray origins for each pixel in the image (Batch_size, 3)
        rays_direction: Ray directions for each pixel in the image (Batch_size, 3)
    Outputs:
        rays_o: Ray origins in NDC space (Batch_size, 3)
        rays_d: Ray directions in NDC space (Batch_size, 3)
    '''
    pass

if __name__ == "__main__":
    H = 4
    W = 4
    focal_length = 1.0
    c2w = torch.ones(4, 4)

    '''
    c2w = torch.tensor([[1/np.sqrt(2), 0, 1/np.sqrt(2), 1], 
                        [0, 1, 0, 1],
                        [-1/np.sqrt(2), 0, 1/np.sqrt(2), 1],
                        [0, 0, 0, 1]])
    '''

    ray_origins, ray_directions = ray_generation(H, W, focal_length, c2w, False)
    print(ray_origins.shape, ray_directions.shape)
    print(ray_origins[0, 0], ray_directions[0, 0])