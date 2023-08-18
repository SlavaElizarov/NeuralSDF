from typing import Literal
import tinycudann as tcnn
from torch import nn
import torch
import torch.nn.functional as F


class Encoding(nn.Module):
    in_features: int
    out_features: int

    def __init__(self):
        super().__init__()


class GridEmbedding(Encoding):
    def __init__(self,
                 in_features: int,
                 num_levels: int = 16,
                 features_per_level: int = 2,
                 log2_hashmap_size: int = 19,
                 base_resolution: int = 16,
                 growth_factor: float = 2.0,
                 use_hash: bool = True,
                 hash_smoothstep: bool = True,
                 mask_k_levels: int = 0,
                 ):
        super().__init__()
        self.in_features = in_features
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.growth_factor = growth_factor
        self.hash_smoothstep = hash_smoothstep
        self.use_hash = use_hash

        assert mask_k_levels < num_levels
        self.mask_k_levels = mask_k_levels

        self.encoding_config = {
            "otype": "HashGrid" if use_hash else "DenseGrid",
            "n_levels": self.num_levels,
            "n_features_per_level": self.features_per_level,
            "log2_hashmap_size": self.log2_hashmap_size,
            "base_resolution": self.base_resolution,
            "per_level_scale": self.growth_factor,
            "interpolation": "Smoothstep" if self.hash_smoothstep else "Linear",
        }

        self.encoding = tcnn.Encoding(
            self.in_features, self.encoding_config, dtype=torch.float32)  # TODO: remove dtype

    @property
    def out_features(self):
        return self.encoding.n_output_dims

    def set_mask_k_levels(self, mask_k_levels: int):
        assert mask_k_levels < self.num_levels
        self.mask_k_levels = mask_k_levels

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # normalize to [0, 1], TODO: remove dirty hack, move it to dataset
        points = points / 3. + 0.5
        features = self.encoding(points)

        if self.mask_k_levels > 0:
            keep_levels = self.num_levels - self.mask_k_levels
            mask = torch.ones(
                (1,) + features.shape[1:], device=features.device, dtype=features.dtype)
            mask[:, keep_levels * self.features_per_level:] = 0.0
            features = features * mask

        return features


class TriplaneEncoding(Encoding):
    def __init__(
        self,
        resolution: int = 32,
        out_features: int = 64,
        reduce: Literal["sum", "product", "concat"] = "sum",
        mask_k_levels: int = 0,
    ) -> None:
        super().__init__()
        self.in_features = 3
        self.out_features = out_features * 3 if reduce == "concat" else out_features
        self.resolution = resolution
        self.reduce = reduce
        self.mask_k_levels = mask_k_levels


        self.planes = nn.Parameter(torch.empty((3,
                                   out_features,
                                   self.resolution, self.resolution)))
        nn.init.xavier_uniform_(self.planes, gain=1.0)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        assert points.shape[-1] == 3

        original_shape = points.shape
        points = points.view(-1, 3) / 1.2

        plane_coord = torch.stack(
            [points[..., [0, 1]], points[..., [0, 2]], points[..., [1, 2]]], dim=0)

        # TODO: should I?
        # Stop gradients from going to sampler
        plane_coord = plane_coord.detach().view(3, -1, 1, 2)

        planes = self.planes
        if self.mask_k_levels > 0:
            factor = 2 ** self.mask_k_levels
            planes = planes.view(
                3, -1, factor, self.resolution // factor, factor, self.resolution // factor)
            planes = torch.mean(planes, dim=(2, 4))

        plane_features = F.grid_sample(
            planes, plane_coord, align_corners=True
        )  # [3, num_components, flattened_bs, 1]

        if self.reduce == "product":
            plane_features = plane_features.prod(
                0).squeeze(-1).T  # [flattened_bs, num_components]
        elif self.reduce == "sum":
            plane_features = plane_features.sum(0).squeeze(-1).T
        elif self.reduce == "concat":
            plane_features = plane_features.permute(2, 0, 1, 3)

        else:
            raise ValueError(f"Unknown reduce method {self.reduce}")

        return plane_features.reshape(*original_shape[:-1], self.out_features)

    def set_mask_k_levels(self, mask_k_levels: int):
        self.mask_k_levels = mask_k_levels

    @torch.no_grad()
    def upsample_grid(self, resolution: int) -> None:
        """Upsamples underlying feature grid

        Args:
            resolution: Target resolution.
        """
        planes = F.interpolate(
            self.planes.data, size=(resolution, resolution), mode="bilinear", align_corners=True
        )

        self.planes = torch.nn.Parameter(planes)
        self.resolution = resolution
