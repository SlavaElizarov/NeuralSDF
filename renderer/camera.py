from typing import Tuple
import torch
from torch import tensor
from pytorch3d.renderer import look_at_view_transform


class Camera:
    def __init__(
        self,
        dist: float = 1.0,
        elev: float = 0.0,
        azim: float = 0.0,
        resolution: int = 256,
        device: str = "cuda",
    ) -> None:
        self.dist = dist
        self.elev = elev
        self.azim = azim
        self.height, self.width = resolution, resolution
        self.device = device

        R, T = look_at_view_transform(
            self.dist, self.elev, self.azim, device=self.device
        )
        self.rotation = R
        self.origin = T

    def set_position(self, dist: float, elev: float, azim: float) -> None:
        self.dist = dist
        self.elev = elev
        self.azim = azim
        R, T = look_at_view_transform(
            self.dist, self.elev, self.azim, device=self.device
        )
        self.rotation = R
        self.origin = T

    def project(self, points: torch.Tensor) -> torch.Tensor:
        """
        Project points to the camera's image plane.
        """
        points = points @ self.rotation[0]  # + self.origin
        return points

    def emit_rays(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Emit rays from the camera's position.
        """

        # generating values between -1 and 1 for each pixel
        xp = torch.arange(0, self.height) / self.height * 2 - 1
        yp = torch.arange(0, self.height) / self.height * 2 - 1

        x, y = torch.meshgrid(xp, yp, indexing="xy")

        # get ray directions
        ray_directions = torch.stack([x, -y, torch.ones_like(x) * -1], dim=-1)
        ray_directions /= torch.linalg.norm(
            ray_directions, dim=-1, keepdim=True
        )  # normalize vectors

        return self.origin, ray_directions.view(-1, 3).to(self.device)
