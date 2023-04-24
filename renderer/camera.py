import math
from typing import Optional, Sequence, Tuple, Union
import torch
from torch.nn import functional as F 


def format_tensor(
    input,
    dtype: torch.dtype = torch.float32,
    device = "cpu",
) -> torch.Tensor:

    if not torch.is_tensor(input):
        input = torch.tensor(input, dtype=dtype, device=device)

    if input.dim() == 0:
        input = input.view(1)

    if input.device == device:
        return input

    input = input.to(device=device)
    return input

def convert_to_tensors_and_broadcast(
    *args,
    dtype: torch.dtype = torch.float32,
    device = "cpu",
):
    # Convert all inputs to tensors with a batch dimension
    args_1d = [format_tensor(c, dtype, device) for c in args]

    # Find broadcast size
    sizes = [c.shape[0] for c in args_1d]
    N = max(sizes)

    args_Nd = []
    for c in args_1d:
        if c.shape[0] != 1 and c.shape[0] != N:
            msg = "Got non-broadcastable sizes %r" % sizes
            raise ValueError(msg)

        # Expand broadcast dim and keep non broadcast dims the same size
        expand_sizes = (N,) + (-1,) * len(c.shape[1:])
        args_Nd.append(c.expand(*expand_sizes))

    return args_Nd

def camera_position_from_spherical_angles(
    distance: float,
    elevation: float,
    azimuth: float,
    degrees: bool = True,
    device = "cpu",
) -> torch.Tensor:

    broadcasted_args = convert_to_tensors_and_broadcast(
        distance, elevation, azimuth, device=device
    )
    dist, elev, azim = broadcasted_args
    if degrees:
        elev = math.pi / 180.0 * elev
        azim = math.pi / 180.0 * azim
    x = dist * torch.cos(elev) * torch.sin(azim)
    y = dist * torch.sin(elev)
    z = dist * torch.cos(elev) * torch.cos(azim)
    camera_position = torch.stack([x, y, z], dim=1)
    if camera_position.dim() == 0:
        camera_position = camera_position.view(1, -1)  # add batch dim.
    return camera_position.view(-1, 3)

def look_at_rotation(
    camera_position, at=((0, 0, 0),), up=((0, 1, 0),), device = "cpu"
) -> torch.Tensor:
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
    # Format input and broadcast
    broadcasted_args = convert_to_tensors_and_broadcast(
        camera_position, at, up, device=device
    )
    camera_position, at, up = broadcasted_args
    for t, n in zip([camera_position, at, up], ["camera_position", "at", "up"]):
        if t.shape[-1] != 3:
            msg = "Expected arg %s to have shape (N, 3); got %r"
            raise ValueError(msg % (n, t.shape))
    z_axis = F.normalize(at - camera_position, eps=1e-5)
    x_axis = F.normalize(torch.cross(up, z_axis, dim=1), eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)
    is_close = torch.isclose(x_axis, torch.tensor(0.0), atol=5e-3).all(
        dim=1, keepdim=True
    )
    if is_close.any():
        replacement = F.normalize(torch.cross(y_axis, z_axis, dim=1), eps=1e-5)
        x_axis = torch.where(is_close, replacement, x_axis)
    R = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)
    return R.transpose(1, 2)


def look_at_view_transform(
    dist = 1.0,
    elev = 0.0,
    azim = 0.0,
    degrees: bool = True,
    eye: Optional[Union[Sequence, torch.Tensor]] = None,
    at=((0, 0, 0),),  # (1, 3)
    up=((0, 1, 0),),  # (1, 3)
    device = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function returns a rotation and translation matrix
    to apply the 'Look At' transformation from world -> view coordinates [0].

    Args:
        dist: distance of the camera from the object
        elev: angle in degrees or radians. This is the angle between the
            vector from the object to the camera, and the horizontal plane y = 0 (xz-plane).
        azim: angle in degrees or radians. The vector from the object to
            the camera is projected onto a horizontal plane y = 0.
            azim is the angle between the projected vector and a
            reference vector at (0, 0, 1) on the reference plane (the horizontal plane).
        dist, elev and azim can be of shape (1), (N).
        degrees: boolean flag to indicate if the elevation and azimuth
            angles are specified in degrees or radians.
        eye: the position of the camera(s) in world coordinates. If eye is not
            None, it will override the camera position derived from dist, elev, azim.
        up: the direction of the x axis in the world coordinate system.
        at: the position of the object(s) in world coordinates.
        eye, up and at can be of shape (1, 3) or (N, 3).

    Returns:
        2-element tuple containing

        - **R**: the rotation to apply to the points to align with the camera.
        - **T**: the translation to apply to the points to align with the camera.

    References:
    [0] https://www.scratchapixel.com
    """

    if eye is not None:
        broadcasted_args = convert_to_tensors_and_broadcast(eye, at, up, device=device)
        eye, at, up = broadcasted_args
        C = eye
    else:
        broadcasted_args = convert_to_tensors_and_broadcast(
            dist, elev, azim, at, up, device=device
        )
        dist, elev, azim, at, up = broadcasted_args
        C = (
            camera_position_from_spherical_angles(
                dist, elev, azim, degrees=degrees, device=device
            )
            + at
        )

    R = look_at_rotation(C, at, up, device=device)
    T = -torch.bmm(R.transpose(1, 2), C[:, :, None])[:, :, 0]
    return R, T

# TODO: rewrite
class Camera:
    def __init__(
        self,
        dist: float = 1.0,
        elev: float = 0.0,
        azim: float = 0.0,
        resolution: int = 256,
        device: str = "cuda",
        half_precision: bool = False,
    ) -> None:
        self.dist = dist
        self.elev = elev
        self.azim = azim
        self.height, self.width = resolution, resolution
        self.device = device
        self.half_precision = half_precision

        R, T = look_at_view_transform(
            self.dist, self.elev, self.azim, device=self.device
        )
        self.rotation = R
        self.origin = T
        
        if self.half_precision:
            self.rotation = self.rotation.half()
            self.origin = self.origin.half()

    def set_position(self, dist: float, elev: float, azim: float) -> None:
        self.dist = dist
        self.elev = elev
        self.azim = azim
        R, T = look_at_view_transform(
            self.dist, self.elev, self.azim, device=self.device
        )
        self.rotation = R #@ torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=R.dtype, device=self.device)
        self.origin = T
        
        if self.half_precision:
            self.rotation = self.rotation.half()
            self.origin = self.origin.half()

    def project(self, points: torch.Tensor) -> torch.Tensor:
        """
        Project points to the camera's image plane.
        """
        points = points @ self.rotation[0]  # + self.origin
        return points

    # TODO: there is a bug here
    # Origin should be the camera position, not the origin
    def emit_rays(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Emit rays from the camera's position.
        """

        # generating values between -1 and 1 for each pixel
        xp = torch.arange(0, self.height, dtype=torch.float16 if self.half_precision else None) / self.height * 2 - 1
        yp = torch.arange(0, self.height, dtype=torch.float16 if self.half_precision else None) / self.height * 2 - 1

        x, y = torch.meshgrid(xp, yp, indexing="xy")

        # get ray directions
        ray_directions = torch.stack([x, -y, torch.ones_like(x) * -1], dim=-1)
        ray_directions /= torch.linalg.norm(
            ray_directions, dim=-1, keepdim=True
        )  # normalize vectors
        

        return self.origin, ray_directions.view(-1, 3).to(self.device)
    
    def get_intrinsic_matrix(self) -> torch.Tensor:
        intrinsic_matrix = torch.zeros((3, 4), dtype=self.rotation.dtype)
        intrinsic_matrix[0, 0] = self.width / 2
        intrinsic_matrix[0, 2] = self.width / 2
        intrinsic_matrix[1, 1] = self.height / 2
        intrinsic_matrix[1, 2] = self.height / 2
        intrinsic_matrix[2, 2] = 1
        return intrinsic_matrix

    def get_extrinsic_matrix(self) -> torch.Tensor:
        """To camera coordinates
        Raises:
            NotImplementedError: _description_
        Returns:
            np.ndarray: _description_
        """
        rotation_matrix = self.rotation
        translation = self.origin

        # translation
        # print(rotation_matrix.shape, camera_position.shape)
        # translation = rotation_matrix @ camera_position

        extrinsic_matrix = torch.zeros((4, 4), dtype=self.rotation.dtype)
        extrinsic_matrix[:3, :3] = rotation_matrix
        extrinsic_matrix[:3, 3] = translation
        extrinsic_matrix[3, 3] = 1.0

        return extrinsic_matrix


