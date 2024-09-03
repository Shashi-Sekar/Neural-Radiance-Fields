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

def hierarchical_sampling(bins: torch.tensor, weights: torch.tensor, num_samples: int, uniform=False):
    '''
    Hierarchical sampling -> to sample more points from those positions we expect to have visual content
    Based on the evaluation of the 'coarse' network on stratified samples
    Parameters:
        bins:           Bins for stratified sampling, shape: (num_rays, num_samples)
        weights:        Weights for the bins, shape: (num_rays, num_samples)
        num_samples:    Number of samples
        uniform:        Whether the samples are equally spaced
    '''
    num_rays, _ = weights.shape

    #Overcome NaN values
    weights = weights + 1e-6

    #Normalize the weights to get the PDF, shape: (num_rays, num_samples)
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)

    #Compute the CDF, shape: (num_rays, num_samples)
    cdf = torch.cumsum(pdf, dim=-1)
    
    #Add zero as the first entry -> CDF is 0 for values < a if the domain is [a, b], shape: (num_rays, num_samples+1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

    #Generate random samples from a uniform distribution (0, 1)
    if uniform:
        #Uniform - Equally spaced samples
        u = torch.linspace(0, 1, num_samples)
        #Matching the shape of CDF expect for the last dimension which should now be the number of samples, shape: (num_rays, num_samples)
        #u = u.expand(list(cdf.shape[:-1]) + [num_samples])
        u = u.expand(num_rays, num_samples)
    else:
        #Random, shape: (num_rays, num_samples)
        #u = torch.rand(list(cdf.shape[:-1]) + [num_samples])
        u = torch.rand(num_rays, num_samples)
    
    #Make the tensor contiguous to avoid memory issues
    u.contiguous()

    #Inverting the CDF
    #Find the first bin where the CDF is greater than the corresponding random sample
    indices = torch.searchsorted(cdf, u, right=True)

    #Avoid out of bounds
    lower_bound = torch.max(torch.zeros_like(indices-1), indices-1)
    higher_bound = torch.min(torch.ones_like(indices)*num_samples, indices)
    indices_bound = torch.stack([lower_bound, higher_bound], dim = -1)
    
    #Matching the the number dimensions of CDF and indices
    #For each ray, each sample, we need the lower and higher bounds
    matched_shape = [indices_bound.shape[0], indices_bound.shape[1], cdf.shape[-1]]
    cdf_bound = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, indices_bound)
    bins_bound = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, indices_bound)

    #Linear Interpolation
    # if u is between lower_cdf and higher_cdf: x = x0 + (x1 - x0) * (u - lower_cdf) / (upper_cdf - lower_cdf)
    # if cdf exists, then u - lower_cdf = 0, so x = x0
    denom = (cdf_bound[..., 1] - cdf_bound[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_bound[..., 0]) / denom
    samples = bins_bound[..., 0] + t * (bins_bound[..., 1] - bins_bound[..., 0])

    return samples

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
    '''
    H = 4
    W = 4
    focal_length = 1.0
    c2w = torch.ones(4, 4)

    c2w = torch.tensor([[1/np.sqrt(2), 0, 1/np.sqrt(2), 1], 
                        [0, 1, 0, 1],
                        [-1/np.sqrt(2), 0, 1/np.sqrt(2), 1],
                        [0, 0, 0, 1]])
    
    ray_origins, ray_directions = ray_generation(H, W, focal_length, c2w, False)
    print(ray_origins.shape, ray_directions.shape)
    print(ray_origins[0, 0], ray_directions[0, 0])
    
    '''
    
    pdf = torch.rand(4,3)
    bins = torch.tensor([1.0, 2.0, 3.0])
    num_samples = 3

    samples = hierarchical_sampling(bins, pdf, num_samples, False)

    print(samples)
