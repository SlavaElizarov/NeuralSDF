from typing import Callable
from renderer.camera import Camera
import torch
from torch import nn
from torch import autograd


class SphereTracingRenderer:
    def __init__(
        self,
        camera: Camera,
        min_dist: float = 0.01,
        max_depth: float = 2,
        max_iteration: int = 30,
    ):
        self.camera = camera
        self.min_dist = min_dist
        self.max_depth = max_depth
        self.max_iteration = max_iteration

    def render(self, sdf: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        origin, directions = self.camera.emit_rays()

        points = torch.zeros_like(directions, requires_grad=False)
        t = torch.zeros(directions.shape[0], dtype=torch.float32).to(self.camera.device)

        is_hit = torch.zeros_like(t).bool()  # is_hit[i] is True if ray i hits a surface
        # condition[i] is True if computation should continue for ray i
        condition = torch.ones_like(t).bool()
        d = torch.zeros_like(t)  # distance to surface for ray i

        with torch.no_grad():

            for _ in range(self.max_iteration):
                d = torch.zeros_like(t)
                points = origin + t[:, None] * directions  # move along ray on t units
                # project points to camera coordinates
                points = self.camera.project(points)
                d[condition] = sdf(points[condition])[:, 0]

                t = t + d

                # check if ray has hit a surface
                is_hit[condition] = torch.abs(d[condition]) < self.min_dist

                # no need to continue if ray has hit a surface
                condition = condition & ~is_hit

                # check if ray has reached maximum depth
                condition = condition & (t < self.max_depth)

                if not condition.any():
                    break

        hit_points = points[is_hit].clone()
        hit_points.requires_grad_(True)
        d = sdf(hit_points)
        (gradient,) = autograd.grad(outputs=d.sum(), inputs=hit_points)

        frame = torch.zeros_like(points)
        frame[is_hit] = (
            gradient / torch.linalg.norm(gradient, dim=-1, keepdim=True)
        ) * 0.5 + 0.5
        return frame.reshape(self.camera.height, self.camera.width, 3)
