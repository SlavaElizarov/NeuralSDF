import math
from typing import Optional
import torch
from torch import Tensor
from torch.nn.functional import pad
from torch.nn import functional as F


def convert_points_from_homogeneous(points: Tensor, eps: float = 1e-8) -> Tensor:
    r"""Function that converts points from homogeneous to Euclidean space.

    Args:
        points: the points to be transformed of shape :math:`(B, N, D)`.
        eps: to avoid division by zero.

    Returns:
        the points in Euclidean space :math:`(B, N, D-1)`.

    Examples:
        >>> input = tensor([[0., 0., 1.]])
        >>> convert_points_from_homogeneous(input)
        tensor([[0., 0.]])
    """
    if not isinstance(points, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(points)}")

    if len(points.shape) < 2:
        raise ValueError(f"Input must be at least a 2D tensor. Got {points.shape}")

    # we check for points at max_val
    z_vec: Tensor = points[..., -1:]

    # set the results of division by zeror/near-zero to 1.0
    # follow the convention of opencv:
    # https://github.com/opencv/opencv/pull/14411/files
    mask: Tensor = torch.abs(z_vec) > eps
    scale = torch.where(mask, 1.0 / (z_vec + eps), torch.ones_like(z_vec))

    return scale * points[..., :-1]


def convert_points_to_homogeneous(points: Tensor) -> Tensor:
    r"""Function that converts points from Euclidean to homogeneous space.

    Args:
        points: the points to be transformed with shape :math:`(*, N, D)`.

    Returns:
        the points in homogeneous coordinates :math:`(*, N, D+1)`.

    Examples:
        >>> input = tensor([[0., 0.]])
        >>> convert_points_to_homogeneous(input)
        tensor([[0., 0., 1.]])
    """
    if not isinstance(points, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(points)}")
    if len(points.shape) < 2:
        raise ValueError(f"Input must be at least a 2D tensor. Got {points.shape}")

    return pad(points, [0, 1], "constant", 1.0)


def get_extrinsic_matrix_from_rotation_and_camera_position(
                                                rotation: Tensor,
                                                camera_position: Tensor,
                                                ) -> torch.Tensor:
    assert rotation.shape[0] == camera_position.shape[0]

    assert rotation.dim() == 3
    assert rotation.shape[1] == 3
    assert rotation.shape[2] == 3

    assert camera_position.dim() == 2
    assert camera_position.shape[1] == 3

    translations = -torch.bmm(rotation, camera_position.unsqueeze(-1)) # (B, 3, 1)
    rotation_translation_matrix = torch.cat([rotation, translations], dim=-1)
    extrinsic_matrix = torch.cat([rotation_translation_matrix, torch.tensor([[[0, 0, 0, 1]]], device=rotation.device).repeat(rotation.shape[0], 1, 1)], dim=1)
    return extrinsic_matrix

def convert_spherical_to_cartesian(
    shperical_coordinates: Tensor,
    degrees: bool = True,
) -> torch.Tensor:
    """Converts spherical coordinates to cartesian coordinates.

    Args:
        shperical_coordinates (Tensor): Tensor of shape (N, 3) containing spherical coordinates. r, theta, phi.
        degrees (bool, optional): Use degrees instead of radians. Defaults to True.

    Returns:
        torch.Tensor: _description_
    """    
    assert shperical_coordinates.dim() == 2
    assert shperical_coordinates.shape[1] == 3

    if degrees:
        shperical_coordinates[:, 1:] = math.pi / 180.0 * shperical_coordinates[:, 1:]

    dist = shperical_coordinates[:, 0]
    elev = shperical_coordinates[:, 1]
    azim = shperical_coordinates[:, 2]

    x = dist * torch.sin(elev) * torch.cos(azim)
    y = dist * torch.sin(elev) * torch.sin(azim)
    z = dist * torch.cos(elev)

    return torch.stack([x, y, z], dim=-1)


def look_at_rotation(
    camera_position: Tensor,
    at_point: Optional[Tensor] = None,
    up_axis: Optional[Tensor] = None
) -> Tensor:
    """
    This function takes a vector 'camera_position' which specifies the location
    of the camera in world coordinates and two vectors `at` and `up` which
    indicate the position of the object and the up directions of the world
    coordinate system respectively. The object is assumed to be centered at
    the origin.

    The output is a rotation matrix representing the transformation
    from world coordinates -> view coordinates.

    Args:
        camera_position: position of the camera in world coordinates
        at: position of the object in world coordinates
        up: vector specifying the up direction in the world coordinate frame.

    The inputs camera_position, at and up can each be a
        - 3 element tuple/list
        - torch tensor of shape (1, 3)
        - torch tensor of shape (N, 3)

    The vectors are broadcast against each other so they all have shape (N, 3).

    Returns:
        R: (N, 3, 3) batched rotation matrices
    """
    assert camera_position.dim() == 2
    assert camera_position.shape[-1] == 3
    
    if up_axis is None:
        up_axis = torch.tensor([[0, 0, 1]], device=camera_position.device, dtype=camera_position.dtype)
    
    if at_point is None:
        at_point = torch.tensor([[0, 0, 0]], device=camera_position.device, dtype=camera_position.dtype)

    z_axis = F.normalize(at_point - camera_position, eps=1e-5)
    x_axis = F.normalize(torch.cross(up_axis, z_axis, dim=1), eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)
    is_close = torch.isclose(x_axis, torch.tensor(0.0, device=camera_position.device, dtype=camera_position.dtype), atol=5e-3).all(
        dim=1, keepdim=True
    )
    if is_close.any():
        replacement = F.normalize(torch.cross(y_axis, z_axis, dim=1), eps=1e-5)
        x_axis = torch.where(is_close, replacement, x_axis)
    R = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)
    return R.transpose(1, 2)

