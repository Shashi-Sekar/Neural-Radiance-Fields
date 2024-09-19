import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os, sys
import time
import json
import random
import imageio
from typing import Tuple

from tqdm import tqdm

from positional_encoding import positional_encoding
from nerf_architecture import NeRF

#Convert the image to 8-bit format
to8b = lambda x : (255*np.clip(x, 0, 1)).astype(np.uint8)

#Convert the output of the model to volume density value and then to alpha 
#This is due to computation of the continuous volumetric rendering using quadrature rule (use discrete set of samples to estimate the integral)
#Alpha is 1 - exp(-sigma * delta) -> sigma is the density, delta is the distance between the samples
# Note: The volume density output by the model needs to be rectified to ensure it is positive
raw2alpha = lambda raw, delta, act_fn=F.relu(): 1.-torch.exp(-act_fn(raw)*delta)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ray_generation(H: int, W: int, focal_length: float, c2w: torch.Tensor) -> Tuple[torch.Tensor, torch.tensor]:
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

def ray_generation_numpy(H: int, W: int, focal_length: float, c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Numpy version of ray generation
    Parameters:
        H: Height of the image
        W: Width of the image
        focal_length: Focal length of the camera
        c2w: Camera to world transformation matrix (4, 4) -> [R | t][0,0,0,1]
    Outputs:
        ray_origins: Ray origins for each pixel in the image (H, W, 3)
        ray_directions: Ray directions for each pixel in the image (H, W, 3)
    '''
    #Get the image coordinates
    i, j = np.meshgrid(torch.arange(W, dtype=np.float32), np.arange(H, dtype=torch.float32), indexing='xy')

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
    ray_directions = np.stack([i, j, -np.ones_like(i)], axis=-1)
    
    #Tranform the ray directions from camera space to world space
    #(W, H, 1, 3) * (3, 3) (Broadcast) -> (W, H, 3, 3) ->(Sum) (W, H, 3)
    ray_directions_world = np.sum(ray_directions[..., np.newaxis, :] * c2w[:3, :3], dim=-1)

    #Ray Origin -> Translation of the camera from the world origin
    #Last column of c2w matrix
    ray_origins = np.broadcast_to(c2w[:3, -1], np.shape(ray_directions_world))

    return ray_origins, ray_directions_world

def ray_generation_given_coordinates(H: int, W: int, focal_length: float, c2w: np.ndarray, coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Generate ray origins and directions for the given coordinates in the image
    Parameters:
        H: Height of the image
        W: Width of the image
        focal_length: Focal length of the camera
        coordinates: Coordinates in the image (N, 2)
    Outputs:
        ray_origins: Ray origins for the given coordinates (N, 3)
        ray_directions: Ray directions for the given coordinates (N, 3)
    '''

    i, j = coordinates[:, 0], coordinates[:, 1]

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
    ray_directions = np.stack([i, j, -np.ones_like(i)], axis=-1)
    
    #Tranform the ray directions from camera space to world space
    #(W, H, 1, 3) * (3, 3) (Broadcast) -> (W, H, 3, 3) ->(Sum) (W, H, 3)
    ray_directions_world = np.sum(ray_directions[..., np.newaxis, :] * c2w[:3, :3], dim=-1)

    #Ray Origin -> Translation of the camera from the world origin
    #Last column of c2w matrix
    ray_origins = np.broadcast_to(c2w[:3, -1], np.shape(ray_directions_world))

    return ray_origins, ray_directions_world

def rays_ndc(H, W, focal, near, rays_origin, rays_direction):
    '''
    Generate the rays in NDC space (Normalized Device Coordinates)
    This is commonly used in triangle rasterization pipeline
    Clipping out points that lie before the near plane
    Parameters:
        H: Height of the image
        W: Width of the image
        focal: Focal length of the camera
        near: Near bound for the scene
        rays_origin: Ray origins for each pixel in the image (Batch_size, 3)
        rays_direction: Ray directions for each pixel in the image (Batch_size, 3)
    Outputs:
        rays_o: Ray origins in NDC space (Batch_size, 3)
        rays_d: Ray directions in NDC space (Batch_size, 3)
    '''
    pass

def importance_sampling(bins: torch.tensor, weights: torch.tensor, num_samples: int, uniform=False):
    '''
    Hierarchical sampling -> to sample more points from those positions we expect to have visual content
    Based on the evaluation of the 'coarse' network on stratified samples
    Parameters:
        bins:           Bins for stratified sampling, shape: (batch_size, num_samples)
        weights:        Weights corresponding to the bins, shape: (batch_size, num_samples)
        num_samples:    Number of samples
        uniform:        Whether the samples are equally spaced
    '''
    batch_size, _ = weights.shape

    #Overcome NaN values
    weights = weights + 1e-6

    #Normalize the weights to get the PDF
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)

    #Compute the CDF
    cdf = torch.cumsum(pdf, dim=-1)
    
    #Add zero for the first entry -> CDF is 0 for values < a if the domain is [a, b], shape: (num_rays, num_samples+1)
    zeros = torch.zeros_like(cdf[..., :1])
    cdf = torch.cat([zeros, cdf], dim=-1)

    #Generate random samples from a uniform distribution (0, 1)
    if uniform:
        #Uniform - Equally spaced samples
        u = torch.linspace(0, 1, num_samples)
        #Matching the shape of CDF expect for the last dimension which should now be the number of samples, shape: (batch_size, num_samples)
        u = u.expand(batch_size, num_samples)
    else:
        #Random, shape: (batch_size, num_samples)
        u = torch.rand(batch_size, num_samples)
    
    #Make sure the tensor is stored in contiguous memory - for use by searchsorted
    u.contiguous()

    #Inverse Transform Samping
    #Piecewise PDF - there could be values in u that are in between two contiguous values in CDF - Need to perform linear interpolation

    #Find the first bin where the CDF is greater than the corresponding random sample, shape: (batch_size, num_samples)
    indices = torch.searchsorted(cdf, u, right=True)

    #Avoid out of bounds
    zeros = torch.zeros_like(indices-1)
    lower_bound = torch.max(zeros, indices-1)
    
    max_index = torch.ones_like(indices)*num_samples
    higher_bound = torch.min(max_index, indices)

    #Stack the lower and higher bounds, shape: (batch_size, num_samples, 2)
    indices_bound = torch.stack([lower_bound, higher_bound], dim = -1)
    
    #Matching the the number dimensions of CDF and indices
    #For each ray, each sample, we need the lower and higher bounds
    matched_shape = [indices_bound.shape[0], indices_bound.shape[1], cdf.shape[-1]]
    cdf_expand = cdf.unsqueeze(1).expand(indices.shape[0], indices.shape[1], cdf.shape[-1])
    cdf_bound = torch.gather(cdf_expand, 2, indices_bound)

    bins_expand = bins.unsqueeze(1).expand(indices.shape[0], indices.shape[1], cdf.shape[-1])
    bins_bound = torch.gather(bins_expand, 2, indices_bound)

    #Linear Interpolation
    # if u is between lower_cdf and higher_cdf: x = x0 + (x1 - x0) * (u - lower_cdf) / (upper_cdf - lower_cdf)
    # if cdf exists, then u - lower_cdf = 0, so x = x0
    denom = (cdf_bound[..., 1] - cdf_bound[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    scale = (u - cdf_bound[..., 0]) / denom
    samples = bins_bound[..., 0] + scale * (bins_bound[..., 1] - bins_bound[..., 0])

    return samples

def stratified_sampling(near, far, num_samples, lindisp=True, jitter=0.):
    '''
    Stratified Sampling
    Parameters:
        near: Near bound for the scene
        far: Far bound for the scene
        num_samples: Number of samples
        lindisp: Whether to sample linearly in inverse depth
        jitter: Stratified sampling
    Outputs:
        samples: Stratified samples (num_rays, num_samples)
    '''
    num_rays = near.shape[0]

    #Sample points between 0 and 1 and then scale them to lie between near and far
            ############    WHY CAN'T DIRECTLY SAMPLE BETWEEN NEAR AND FAR?    ############
    u = torch.linspace(0., 1., steps=num_samples)
    if lindisp:
        z_vals = 1./(1./near * (1-u) + 1./far * u)
    else:
        z_vals = near*(1-u) + far*u

    z_vals = z_vals.expand(num_rays, num_samples)

    if jitter > 0.:
        #need to jitter around the midpoints of the intervals adding non-uniformity
        right = z_vals[..., 1:]
        left = z_vals[..., :-1]
        mid = (left + right) * 0.5

        upper = torch.cat([mid, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mid], dim=-1)
        t_rand = torch.rand(z_vals.shape).to(device)
        z_vals = lower + t_rand * (upper - lower)

    return z_vals

def sigma_loss(rays_o, rays_d, viewdirs, near, far, depths, run_func, network, N_samples, perturb, raw_noise_std, err=1):
    '''
    Depth Supervised Loss and Color Loss
    Parameters:
        rays_o: Ray origins for each pixel in the image (batch_size, 3)
        rays_d: Ray directions for each pixel in the image (batch_size, 3)
        viewdirs: View directions (batch_size, 3)
        near: Near bound for the scene (batch_size)
        far: Far bound for the scene (batch_size)
        depths: Depth values (batch_size)
        run_func: Function to pass queries to the model
        network: Model to predict the RGB and density at each sampled point
        N_samples: Number of samples
        perturb: Perturb the depth values
        raw_noise_std: Raw noise standard deviation
        err: Error threshold
    '''

    num_rays = rays_o.shape[0]
    t_vals = torch.linspace(0., 1., N_samples).to(device)
    t_vals = t_vals.expand(num_rays, N_samples)

    #Stratified Sampling
    # (num_rays, num_samples)
    z_vals = stratified_sampling(near, far, N_samples, True, 0.)

    #Samples along the ray direction and from the ray origin
    # (num_rays, num_samples, 3)
    sampled_pts = rays_o[...,np.newaxis,:] + rays_d[...,np.newaxis,:] * z_vals[..., :, np.newaxis]
    
    raw = run_func(sampled_pts, viewdirs, network)

    noise = 0.
    if raw_noise_std > 0.1:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std
    
    #Calculate the distance between adjacent samples
    # (num_rays, num_samples-1)
    delta = z_vals[...,1:] - z_vals[...,:-1]

    #Distance from the last sample is infinity
    # (num_rays, num_samples)
    delta = torch.cat([delta, torch.tensor([1e10]).to(device).expand(delta[...,:1].shape)], -1)

    #Multiply the distances by the norm of ray direction to scale it, (num_rays, num_samples)
    delta = delta * torch.norm(rays_d[..., None,:], dim=-1)

    #Volume Density, (num_ray, num_samples)
    alpha = raw2alpha(raw[..., 3] + noise, delta)

    #Ray termination distribution, 'h'
    #Calculate the accumulated transmittance using alpha
    # (num_rays, num_samples)
    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(device), 1.-alpha + 1e-10], dim=-1), dim=-1)[:,:-1]
    weights = alpha * T

    #Depth Loss
    loss = -torch.log(weights + 1e-5) * torch.exp(-(z_vals - depths[:,None]) ** 2 / (2 * err)) * delta
    loss = torch.sum(loss, -1)

    return loss

def batchify_rays(rays_info, batch_size):
    '''
    Batch the rays and perform volumetric rendering
    Parameters: 
        rays_info: Ray information -> origin, direction, near, far, viewdirs, depth (if true) (batch_size, 3+3+1+1+3+1)
        batch_size: Batch size for processing rays
    Outputs:
        rgb_map: Predicted RGB values (batch_size, 3)
        inv_depth_map: Inverse Depth map (batch_size)
        acc_map: Accumulated Alpha Map (batch_size)
        depth_map: Depth map (batch_size)
    '''

    total_rays = rays_info.shape[0]
    all_output = {}

    for i in range(0, total_rays, batch_size):
        rays_batch = volumetric_rendering(rays_info[i:i+batch_size])
        for k in rays_batch:
            if k not in all_output:
                all_output[k] = []
            all_output[k].append(rays_batch[k])
    
    all_output = {k: torch.cat(all_output[k], dim=0) for k in all_output}

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_background=False, pytest=False):
    '''
    Mapping the raw predictions to the final outputs
    Parameters:
        raw: Predictions, (batch_size, num_samples, 4)
        z_vals: Stratified samples, (batch_size, num_samples)
        rays_d: Ray directions (batch_size, 3)
    Outputs:
        rgb_map: Predicted RGB values (batch_size, 3)
        inv_depth_map: Inverse Depth map (batch_size)
        acc_map: Sum of weights (batch_size)
        weights: Weights corresponding to each sampled color (batch_size, num_samples)
        depth_map: Estimated distance to object (batch_size)
    '''

    #Calculate the distance between adjacent samples
    # (num_rays, num_samples-1)
    delta = z_vals[..., 1:] - z_vals[..., :-1]

    #Distance from the last sample is infinity
    # (num_rays, num_samples)
    delta = torch.cat([delta, torch.tensor([1e10]).to(device).expand(delta[...,:1])], dim=-1)

    #Multiply the distances by the norm of ray direction to scale it, (batch_size, num_samples)
    delta = delta * torch.norm(rays_d[..., np.newaxis, :], dim=-1)

    #Extract the RGB values, (num_rays, num_samples, 3)
    rgb = raw[..., :3]
    
    #Adding noise to the prediction. Acts as a regularizer
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., :3].shape) * raw_noise_std
        noise = noise.to(device)

    #Volume Density, (num_rays, num_samples)
    alpha = raw2alpha(raw[..., 3] + noise, delta)
    
    #Ray termination distribution, 'h'
    #Computing the weights, (num_rays, num_samples)
    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(device), 1.-alpha + 1e-10], dim=-1), dim=-1)[:,:-1]
    weights = alpha * T

    #Compute the RGB Map, (num_rays, 3)
    rgb_map = torch.sum(weights[..., np.newaxis] * rgb, dim=-2)

    #Depth Map is the expected distance, (num_rays)
    depth_map = torch(weights * z_vals, dim=-1)
    
    #Disparity is inverse depth
    inv_depth_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, dim=-1))

    #Accumulated weights, (num_rays)
    acc_map = torch.sum(weights, dim=-1)

    if white_background:
        rgb_map = rgb_map + (1.-acc_map[..., np.newaxis])

    return rgb_map, inv_depth_map, acc_map, weights, depth_map

def render(H, W, focal, batch_size=1024*32, rays=None, c2w=None, ndc=True, near=0., far=1., use_viewdirs=False, c2w_staticcam=None, depth=None):
    '''
    Following operations
        1. Ray Generation
        2. Batch them
        3. Volumetric Rendering
    Parameters:
        H: Height of the image (pixels)
        W: Width of the image (pixels)
        focal: Focal length of the camera
        batch_size: Batch size for processing rays
        rays: Ray origins and directions for each pixel in the batch (2, batch_size, 6)
        c2w: Camera to world transformation matrix (4, 4) -> [R | t][0,0,0,1]
        ndc: Whether to use NDC space (if True, origin and direction needs to be in NDC)
        near: Near bound for the scene (batch_size)
        far: Far bound for the scene (batch_size)
        use_viewdirs: Whether to use the view directions
        c2w_staticcam: Camera to world transformation matrix for the static camera (4, 4) -> [R | t][0,0,0,1]. Use this for camera
        depth: 
    Outputs:
        rgb_map: Predicted RGB values (batch_size, 3)
        inv_depth_map: Inverse Depth map (batch_size)
        acc_map: Sum of weights (batch_size)
        depth_map: Depth map (batch_size)
    '''

    rays_o, rays_d = ray_generation(H, W, focal, c2w)

    shape = rays_o.shape

    #Unitize the view directions
    viewdirs = rays_d
    viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)

    #Flatten
    viewdirs = viewdirs.view(-1, 3).float()
    rays_o = rays_o.view(-1, 3).float()
    rays_d = rays_d.view(-1, 3).float()

    #Set the near and far planes
    near = near * torch.ones(rays_d[...,:1])
    far = far * torch.ones(rays_d[...,:1])

    #Concatenating ray information
    rays_info = torch.cat([rays_o, rays_d, near, far, viewdirs], dim=-1)
    
    if depth is not None:
        rays_info = torch.cat([rays_info, depth], dim=-1)

    #Batch the rays
    all_output = batchify_rays(rays_info, batch_size)
    for k in all_output:
        k_shape = list(shape[:-1]) + list(all_output.shape[1:])
        all_output[k] = all_output[k].view(k_shape)

    k_extract = ['rgb_map', 'inv_depth_map', 'acc_map', 'depth_map']
    ret_list = [all_output[k] for k in k_extract]
    ret_dict = {k: all_output[k] for k in all_output if k not in k_extract}

    return ret_list + [ret_dict]

def volumetric_rendering(rays_info, network_fn, network_query_fn, N_samples, retraw=False, lindisp=False, jitter=0., N_importance=0, network_fine=None, white_background=False, raw_noise_std=0., verbose=False, pytest=False, sigma_loss=None):
    '''
    Volumetric Rendering
        1. Stratified Sampling
        2. 
    Parameters:
        rays_info: Ray information -> origin, direction, near, far, viewdirs, depth (if true) (batch_size, 3+3+1+1+3+1)
        network_fn: Model to predict the RGB and density at each sampled point
        network_query_fn: Function to pass queries to the model
        N_samples: Number of samples
        retraw: if True, return raw predictions
        lindisp: if True, sample linearly in inverse depth
        jitter: Stratified sampling
        N_importance: Number of samples for importance sampling
        network_fine: Fine network same spec as network_fn
        white_background: White background
        raw_noise_std: Raw noise standard deviation
        verbose: Print debugging information
        pytest: Pytest mode
        sigma_loss: Sigma loss
    Outputs:
        rgb_map: Predicted RGB values (batch_size, 3)
        inv_depth_map: Inverse Depth map (batch_size)
        acc_map: Sum of weights (batch_size)
        raw: Raw predictions (batch_size, num_samples, 4)
        rgb0: RGB output of coarse model (batch_size, 3)
        inv_depth0: Inverse Depth output of coarse model (batch_size)
        acc0: Sum of weights of coarse model (batch_size)
        z_std: Standard Deviation of distances along ray for each sample (batch_size)
    '''
    batch_size = rays_info.shape[0]
    
    rays_o, rays_d, near, far, viewdirs = rays_info[:,:3], rays_info[:,3:6], rays_info[:,6], rays_info[:,7], rays_info[:,8:11]
    near = near.unsqueeze(-1) #(batch_size, 1)
    far = far.unsqueeze(-1) #(batch_size, 1)
    
    z_vals = stratified_sampling(near, far, N_samples, lindisp, jitter)
    
    #Samples along the ray direction and from the ray origin
    sampled_pts = rays_o[...,np.newaxis,:] + rays_d[...,np.newaxis,:] * z_vals[..., :, np.newaxis]

    raw = network_query_fn(sampled_pts, viewdirs, network_fn)
    rgb_map, inv_depth_map, acc_map, weights, depth_map= raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_background, pytest)

    #Importance Sampling
    if N_importance > 0:
        rgb0, inv_depth0, acc0 = rgb_map, inv_depth_map, acc_map

        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = importance_sampling(z_vals_mid, weights[..., 1:-1], N_importance, (jitter==0.))
        z_samples = z_samples.detach()

        #Combine samples from coarse and fine networks
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)
        sampled_pts = rays_o[...,np.newaxis,:] + rays_d[...,np.newaxis,:] * z_vals[..., :, np.newaxis]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(sampled_pts, viewdirs, run_fn)
        rgb_map, inv_depth_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_background, pytest)

    ret = {'rgb_map': rgb_map, 'inv_depth_map': inv_depth_map, 'acc_map': acc_map, 'raw': raw, 'depth_map': depth_map}

    if retraw:
        ret['raw'] = raw
    
    if N_importance > 0:
        ret['rgb0'] = rgb0
        ret['inv_depth0'] = inv_depth0
        ret['acc0'] = acc0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)

    if sigma_loss is not None and rays_info.shape[-1] > 11:
        depths = rays_info[:, 11]
        ret['sigma_loss'] = sigma_loss.calculate_loss(rays_o, rays_d, viewdirs, near, far, depths, network_query_fn, network_fine)

    for k in ret:
        if torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any():
            print(f"NaN or Inf in {k}")

    return ret

def render_path(H, W, focal_length, render_poses, batch_size, savedir=None, render_factor=0):
    '''
    Render the scene for the given set of poses
    Parameters:
        H: Height of the image
        W: Width of the image
        focal_length: Focal length of the camera
        render_poses: Poses to render the scene
        batch_size: Batch size for processing rays
        savedir: Directory to save the rendered images
        render_factor: Downsample factor to increase speed
    Outputs:
        rgb_map: List of Predicted RGB values (number of poses, batch_size, 3)
        inv_depth_map: List of Inverse Depth map (number of poses, batch_size)
    '''

    #Downsample the image for faster rendering
    if render_factor > 0:
        H = H // render_factor
        W = W // render_factor
        focal_length = focal_length / render_factor

    rgb_list = []
    inverse_depth_list = []

    curr_time = time.time()

    for i, c2w in enumerate(tqdm(render_poses)):
        print(f"Rendering pose {i+1}/{len(render_poses)},\tTime: {time.time()-curr_time:.4f}")
        curr_time = time.time()

        rgb, inv_depth, acc_map, depth, _ = render(H, W, focal_length, batch_size, c2w=c2w[:3,:4], ndc=False)

        rgb_list.append(rgb.cpu().numpy())
        inverse_depth_list.append(inv_depth.cpu().numpy())

        #Save the rendered images
        if savedir is not None:
            #RGB
            rgb8 = to8b(rgb.cpu().numpy())
            rgb8[np.isnan(rgb8)] = 0
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

            #Depth
            depth = depth.cpu().numpy()
            filename = os.path.join(savedir, '{:03d}_depth.png'.format(i))
            imageio.imwrite(filename, depth)

            np.savez(os.path.join(savedir, '{:03d}.npz'.format(i)), rgb=rgb_list[-1], inv_depth=inverse_depth_list[-1], acc_map=acc_map.cpu().numpu(), depth=depth)

    rgb_list = np.stack(rgb_list, axis=0)
    inverse_depth_list = np.stack(inverse_depth_list, axis=0)

    return rgb_list, inverse_depth_list
    
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

    samples = importance_sampling(bins, pdf, num_samples, False)

    print(samples)
