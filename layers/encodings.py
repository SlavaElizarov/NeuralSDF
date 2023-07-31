import tinycudann as tcnn
from torch import nn
import torch


class GridEmbedding(nn.Module):
    def __init__(self,         
                in_features: int,
                num_levels: int = 16,
                features_per_level: int = 2,
                log2_hashmap_size: int = 19,
                base_resolution: int = 16,
                per_level_scale: float = 2.0,
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
        self.growth_factor = per_level_scale
        self.hash_smoothstep = hash_smoothstep
        self.use_hash = use_hash

        assert mask_k_levels < num_levels
        self.mask_k_levels = mask_k_levels

        self.encoding_config={
            "otype": "HashGrid" if use_hash else "DenseGrid",
            "n_levels": self.num_levels,
            "n_features_per_level": self.features_per_level,
            "log2_hashmap_size": self.log2_hashmap_size,
            "base_resolution": self.base_resolution,
            "per_level_scale": self.growth_factor,
            "interpolation": "Smoothstep" if self.hash_smoothstep else "Linear",
        }
        
        self.encoding = tcnn.Encoding(self.in_features, self.encoding_config, dtype=torch.float32)  # TODO: remove dtype

    @property
    def out_features(self):
        return self.encoding.n_output_dims
    
    def set_mask_k_levels(self, mask_k_levels: int):
        assert mask_k_levels < self.num_levels
        self.mask_k_levels = mask_k_levels

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        points = points / 3. + 0.5 # normalize to [0, 1], TODO: remove dirty hack, move it to dataset
        features = self.encoding(points)

        if self.mask_k_levels > 0:
            keep_levels = self.num_levels - self.mask_k_levels
            mask = torch.ones((1,) + features.shape[1:], device=features.device, dtype=features.dtype)
            mask[:, keep_levels * self.features_per_level:] = 0.0
            features = features * mask

        return features

