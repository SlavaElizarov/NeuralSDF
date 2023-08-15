from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Cameras(nn.Module):
    def __init__(self,
                 rotation: Tensor,
                 camera_position: Tensor,
                 focal_length: Tensor,
                 height: int,
                 width: int,
                 requires_grad: bool = False,
                 device="cpu"):
        """ Fully differentiable perspective camera class.
        https://www.youtube.com/watch?v=qByYk6JggQU

        Can be used to transform points from world space to screen space.
        Emit rays from camera origin to screen space for ray/sphere tracing and NERFs.

        Args:
            rotation (Tensor): camera rotation matrices in world space of shape (B, 3, 3).
            camera_position (Tensor): camera positions in world space of shape (B, 3).
            focal_length (Tensor): focal length of shape (B).
            height (int): height of the image.
            width (int): width of the image.
            requires_grad (bool, optional): requires_grad Defaults to False.
            device (str, optional): device Defaults to "cpu".
        """        

        super().__init__()
        assert rotation.dim() == 3
        assert rotation.shape[-2:] == (3, 3)
        assert camera_position.dim() == 2
        assert camera_position.shape[-1] == 3
        assert focal_length.dim() == 1

        self.rotation = nn.Parameter(rotation.to(
            device), requires_grad=requires_grad)
        self.camera_position = nn.Parameter(
            camera_position.to(device), requires_grad=requires_grad)
        self.focal_length = nn.Parameter(
            focal_length.to(device), requires_grad=requires_grad)

        self.height = height
        self.width = width
        self.device = device

    def get_calibration_matrices(self) -> Tensor:
        r"""Return the calibration matrices of the storage.

        Returns:
            tensor of shape (B, 3, 3) containing the calibration matrices.
        """
        intrinsic = torch.eye(3, device=self.device, dtype=self.rotation.dtype).unsqueeze(
            0).repeat(self.batch_size, 1, 1)
        intrinsic[:, 0, 0] = self.focal_length * self.width / 2
        intrinsic[:, 1, 1] = self.focal_length * self.height / 2
        intrinsic[:, 0, 2] = self.width / 2
        intrinsic[:, 1, 2] = self.height / 2
        return intrinsic  # Bx4x4

    @property
    def batch_size(self) -> int:
        """Return the batch size of the storage.

        Returns:
            scalar with the batch size.
        """
        return self.rotation.shape[0]

    
    def forward(self, points: Tensor, camera_indices: Tensor) -> Tensor:
        """Transforms points from world space to screen space.

        Args:
            points (Tensor): tensor of shape (N, 3) containing points in world space.
            camera_indices (Tensor): tensor of shape (N) containing the camera indices for each point.

        Returns:
            Tensor: tensor of shape (N, 2) containing points in screen space.
        """        
        assert points.dim() == 3
        assert points.shape[-1] == 3
        assert camera_indices.dim() == 1
        assert camera_indices.shape[0] == points.shape[0]

        projection = self.projection[camera_indices].unsqueeze(1) # Bx1x4x4

        # to homogeneous
        points_h = convert_points_to_homogeneous(points)  # BxNx4
        # transform coordinates
        points_0_h = torch.matmul(points_h, projection.permute(0, 2, 1)) # BxNx4
        # points_0_h = torch.bmm(points_h, projection.permute(0, 2, 1))
        points_0_h = torch.squeeze(points_0_h, dim=-1)
        # to euclidean
        return convert_points_from_homogeneous(points_0_h)  # BxNx3
    
    
    def emit_rays(self,
                  camera_indices: Optional[Tensor] = None,
                  rays_per_pixel: int = 1,
                  uv: Optional[Tensor] = None) -> Tensor:
        """Emit rays from camera origin to screen space for ray/sphere tracing and NERFs.

        Args:
            camera_indices (Tensor): tensor of shape (N) containing the camera indices for each point. (B)
            coords (Optional[Tensor], optional): UV coordinates of ray origins. Defaults to None. (B, N, 2)

        Returns:
            Tensor: tensor of shape (N, 3) containing ray directions.
        """
        return self.rotation.shape[0]

    def forward(self, points: Tensor, camera_indices: Tensor) -> Tensor:
        """Transforms points from world space to screen space.

        Args:
            points (Tensor): tensor of shape (B, N, 3) containing points in world space.
            camera_indices (Tensor): tensor of shape (B).

        Returns:
            Tensor: tensor of shape (N, 2) containing points in screen space.
        """
        assert points.shape[-1] == 3
        assert camera_indices.dim() == 1
        assert camera_indices.shape[0] == points.shape[0]

        rotation = self.rotation[camera_indices]  # Bx3x3
        camera_position = self.camera_position[camera_indices].unsqueeze(
            1)  # Bx1x3

        # to camera space
        points = points - camera_position  # BxNx3
        points = torch.matmul(points, rotation.permute(0, 2, 1))  # BxNx3

        # to screen space
        intrinsic = self.get_calibration_matrices()[camera_indices]  # Bx3x3
        points = torch.matmul(points, intrinsic.permute(0, 2, 1))  # BxNx3
        uv = points[..., :2] / points[..., 2:]  # BxNx2

        # flip v axis
        uv[..., 1] = self.height - uv[..., 1]

        return uv, points[..., -1]

    def emit_rays(self,
                  camera_indices: Optional[Tensor] = None,
                  uv: Optional[Tensor] = None) -> Tensor:
        """Emit rays from camera origin to screen space for ray/sphere tracing and NERFs.

        Args:
            camera_indices (Tensor): tensor of shape (N) containing the camera indices for each point. (B)
            coords (Optional[Tensor], optional): UV coordinates of ray origins. Defaults to None. (B, N, 2)

        Returns:
            Tensor: tensor of shape (N, 3) containing ray directions.
        """
        assert camera_indices is None or camera_indices.dim() == 1
        assert uv is None or uv.dim() == 3
        assert uv is None or uv.shape[0] == camera_indices.shape[0]

        if camera_indices is None:
            camera_indices = torch.arange(
                self.batch_size, device=self.device, dtype=torch.long)

        if uv is None:
            # generating values between -1 and 1 for each pixel
            max_size = max(self.height, self.width)
            xp = torch.linspace(0, self.width, self.width,
                                dtype=self.rotation.dtype) / max_size * 2 - (self.width / max_size)
            yp = torch.linspace(0, self.height, self.height,
                                dtype=self.rotation.dtype) / max_size * 2 - (self.height / max_size)
            x, y = torch.meshgrid(xp, yp, indexing="xy")
            uv = torch.stack([x, -y], dim=-1).reshape(1, -1, 2)
            uv = uv.to(self.device)

        canvas = torch.cat([uv, torch.ones_like(
            uv[..., :1], device=self.device) * self.focal_length.view(-1, 1, 1)], dim=-1)  # BxNx3
        canvas = torch.matmul(canvas, self.rotation[camera_indices])  # BxNx3
        ray_directions = F.normalize(canvas, dim=-1)  # BxNx3
        origins = self.camera_position[camera_indices].unsqueeze(1)  # Bx1x3
        origins = origins + canvas * 0.0  # BxNx3

        return ray_directions, origins
