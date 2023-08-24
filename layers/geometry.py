from typing import Union
from renderer.camera import Cameras
import torch
from torch import nn

def make_3d_grid(resolution: int, device: Union[torch.device, str] = "cuda") -> torch.Tensor:
    """
    Creates a uniform 3D grid of size resolution x resolution x resolution.
    Args:
        resolution (int): resolution of the grid
        device (torch.device): device to create the grid on

    Returns:
        torch.Tensor: 3D grid, tensor of size (resolution x resolution x resolution x 3)
    """
    with torch.no_grad():
        r = torch.linspace(-1, 1, resolution, device=device) # [-1, 1]
        x, y, z = torch.meshgrid(r, r, r, indexing='ij')
        grid = torch.stack((x, y, z), dim=-1)
        return grid


class FeatureVolumeEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, points: torch.Tensor, feature_volume: torch.Tensor) -> torch.Tensor:
        """ Get features at the points using a dense feature volume (cube)


        Args:
            points (torch.Tensor): coordinates of the points (B, M, 3) coordinates are in the range [-1, 1]
            feature_volume (torch.Tensor): dense feature volume (B, C, resolution, resolution, resolution)

        Returns:
            torch.Tensor: features at the points (B, C, M)
        """

        assert points.dim() == 3, "Points must be a 2D tensor"
        assert points.shape[-1] == 3, "Points must have 3 coordinates"
        assert feature_volume.dim() == 5, "Feature volume must be a 5D tensor"
        assert feature_volume.shape[2] == feature_volume.shape[3] == feature_volume.shape[4], "Feature volume must be a cube"

        # Get batch size
        batch_size = points.shape[0]

        # Get the features at the points
        points = points.view(batch_size, -1, 1, 1, 3) # (B, M, 1, 1, 3)
        features = nn.functional.grid_sample(feature_volume, points, align_corners=True, padding_mode="border") # (B, C, M, 1, 1)
        
        # Reshape the features
        features = features.view(batch_size, *features.shape[1:-2]) # (B, C, M)

        return features


class FeatureVolumeLayer(nn.Module):
    def __init__(self, resolution: int, device: Union[torch.device, str] = "cuda"):
        super().__init__()
        self._resolution = resolution
        self._device = device
        self._grid = make_3d_grid(resolution, device=device)

    def forward(self, cameras: Cameras, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cameras (Cameras): camera object (B*V)
            features (torch.Tensor): tensor of size (B, V, C, H, W)

        Returns:
            torch.Tensor: tensor of size (B, C, resolution, resolution, resolution)
        """
        assert features.dim() == 5, "Features must be a 5D tensor"

        # Get batch size and number of views
        batch_size = features.shape[0] # B
        num_views = features.shape[1] # V

        # Project the grid points into each camera
        coordinates = self._grid.view(1, -1, 3) # (1, M, 3) where M = resolution^3
        indices = torch.arange(batch_size * num_views, device=self._device)
        uv, _ = cameras.forward(coordinates, indices, normalize_uv=True) # (B * V, M, 2)

        # Get mask of visible points
        mask = (uv[..., 0] >= 0) & (uv[..., 0] <= 1) & (uv[..., 1] >= 0) & (uv[..., 1] <= 1)  # (B * V, M)
        # TODO: Consider to mask out points with unsufficient number of corresponding views (e.g. 2)

        # Get the features at the projected points
        features = features.view(-1, *features.shape[2:]) # (B*V, C, H, W)
        features = nn.functional.grid_sample(features, uv.view(batch_size * num_views, -1, 1, 2),
                                             align_corners=True, 
                                             padding_mode="zeros") # (B*V, C, M, 1)

        # Reshape the features
        features = features.view(batch_size, num_views, *features.shape[1:-1]) # (B, V, C, M)
        mask = mask.view(batch_size, num_views, -1) # (B, V, M)

     
        # Aggregate the mask
        counts = mask.sum(dim=1, dtype=features.dtype) # (B, M)
        mask = counts > 0 # (B, M)

        # Get the mean of the features
        features_mean = features.sum(dim=1) / counts.unsqueeze(-2) # (B, C, M)

        # Get variance of the features
        features_second_moment = features.pow(2).sum(dim=1) / counts.unsqueeze(-2) # (B, C, M)
        features_variance = features_second_moment - features_mean.pow(2) # (B, C, M)

        # Concatenate mean and variance
        features = torch.cat((features_mean, features_variance), dim=1) # (B, 2*C, M)
        
        return features, coordinates, mask
