from abc import ABC
import os
from typing import Tuple
import numpy as np

import torch
from torch.utils.data import Dataset
from pytorch3d.structures import Meshes
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes

from utils import LatinHypercubeSampler


class MeshSampler(ABC):
    def __init__(self, mesh: Meshes):
        self.mesh = mesh

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def __call__(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sample(num_samples)


class UniformMeshSampler(MeshSampler):
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        samples = sample_points_from_meshes(
            self.mesh, num_samples=num_samples, return_normals=True
        )
        assert len(samples) == 2
        points, normals = samples
        return points, normals


class MeshDataset(Dataset):
    def __init__(
        self,
        mesh_path: str,
        samples_per_epoch: int = 1000,
        add_vertices: bool = True,
        device: str = "cuda",
    ):
        assert os.path.exists(mesh_path)
        assert mesh_path.endswith(".obj")
        assert samples_per_epoch > 0
        assert device in ["cuda", "cpu"]

        self.device = device
        self.mesh_filepath = mesh_path
        self.samples_per_epoch = samples_per_epoch
        self.add_vertices = add_vertices
        self.mesh: Meshes = self._load_mesh(mesh_path)
        self.mesh_sampler = UniformMeshSampler(self.mesh)
        self.space_sampler = LatinHypercubeSampler(
            np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
        )

        self._surface_points: torch.Tensor
        self._normals: torch.Tensor
        self._off_surface_points: torch.Tensor
        self.resample()

    def _load_mesh(self, mesh_filepath: str):
        meshes = load_objs_as_meshes([mesh_filepath], device=self.device)
        return self._normalize_to_unit_sphere(meshes)

    def resample(self):
        points, normals = self.mesh_sampler(self.samples_per_epoch)
        points, normals = points[0], normals[0]
        if self.add_vertices:
            points = torch.cat([points, self.mesh.verts_packed()], dim=0)  # type: ignore
            normals = torch.cat([normals, self.mesh.verts_normals_packed()], dim=0)  # type: ignore

        self._surface_points = points
        self._normals = normals
        self._off_surface_points = self.space_sampler(
            self._surface_points.shape[0], device=self.device
        )

    def _normalize_to_unit_sphere(self, meshes: Meshes) -> Meshes:
        V: torch.Tensor = meshes.verts_packed()  # type: ignore
        V_max, _ = torch.max(V, dim=0)
        V_min, _ = torch.min(V, dim=0)
        V_center = (V_max + V_min) / 2.0
        meshes.offset_verts_(V_center * -1)

        V = meshes.verts_packed()  # type: ignore

        # Find the max distance to origin
        max_dist = torch.sqrt(torch.max(torch.sum(V**2, dim=-1)))
        V_scale = 1.0 / max_dist
        meshes.scale_verts_(V_scale.item())

        return meshes

    def __len__(self):
        return self._surface_points.shape[0]

    def __getitem__(self, idx):
        return (
            self._surface_points[idx],
            self._normals[idx],
            self._off_surface_points[
                idx % self._off_surface_points.shape[0]
            ],  # TODO: this is a hack
        )
