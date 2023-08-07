
from models.sdf import SDF
from renderer.camera import Cameras
import torch



class SphereTracingRenderer:
    def __init__(
        self,
        camera: Cameras,
        min_dist: float = 0.01,
        max_depth: float = 2,
        max_iteration: int = 30,
    ):
        self.camera = camera
        self.min_dist = min_dist
        self.max_depth = max_depth
        self.max_iteration = max_iteration

    def render(self, sdf: SDF) -> torch.Tensor:
        directions, origin = self.camera.emit_rays()
        directions = directions.reshape(-1, 3)
        origin = origin.reshape(-1, 3)

        t = torch.zeros(directions.shape[0], dtype=directions.dtype).to(self.camera.device)

        is_hit = torch.zeros_like(t).bool()  # is_hit[i] is True if ray i hits a surface
        # condition[i] is True if computation should continue for ray i
        condition = torch.ones_like(t).bool()
        d = torch.zeros_like(t)  # distance to surface for ray i


        with torch.no_grad():

            for _ in range(self.max_iteration):
                d = torch.zeros_like(t)
                points = origin + t[:, None] * directions  # move along ray on t units
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
        frame = torch.zeros_like(points)
        if is_hit.any():
            hit_points = points[is_hit].clone()
            hit_points.requires_grad_(True)
            _, gradient = sdf.forward_with_grad(hit_points)
            
            normals = gradient / torch.linalg.norm(gradient, dim=-1, keepdim=True)
            # TODO: fix camera
            color = torch.einsum('ik,ik->i', normals, -directions[is_hit].view(-1, 3))
            
            frame[is_hit] = torch.stack([color]*3, dim=-1)
            frame = frame.reshape(self.camera.height, self.camera.width, 3)
        return frame

