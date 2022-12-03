import os

import torch
from torch.utils.data import Dataset
from pytorch3d.structures import Meshes
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes


class MeshDataset(Dataset):
    def __init__(
        self, mesh_filepath: str, frac_points_to_sample: int = 0, device: str = "cuda"
    ):
        assert os.path.exists(mesh_filepath)
        assert mesh_filepath.endswith(".obj")

        meshes = load_objs_as_meshes([mesh_filepath], device=device)
        self.mesh: Meshes = self.normalize_to_unit_sphere(meshes)
        self.frac_points_to_sample = frac_points_to_sample

        self.vertices: torch.Tensor
        self.normals: torch.Tensor
        self.set_vertices_and_normals()
        self.add_random_points()

        self.steps_til_resample = self.__len__()

    def set_vertices_and_normals(self):
        vertices = self.mesh.verts_packed()
        assert vertices is not None
        self.vertices = vertices

        normals = self.mesh.verts_normals_packed()
        assert normals is not None
        self.normals = normals

    def add_random_points(self) -> None:
        n_points_to_sample = int(self.vertices.shape[0] * self.frac_points_to_sample)

        if n_points_to_sample <= 0:
            return

        # There is a problem with such strategy since sample weight is proportional to face area
        # So biggest faces are sampled more than smaller ones it can cause a lack of details
        # (complex geometry modeled with a smalest faces)
        # We should sample points evenly. There is a sutable function in trimesh trimesh.sample.sample_surface_even
        # Another option reveight regions based on curvature
        # TODO: investigate sampling strategies
        samples = sample_points_from_meshes(
            self.mesh, n_points_to_sample, return_normals=True
        )
        assert len(samples) == 2
        sampled_points, sampled_normals = samples

        self.vertices = torch.cat([self.vertices, sampled_points[0]], dim=0)
        self.normals = torch.cat([self.normals, sampled_normals[0]], dim=0)

    def normalize_to_unit_sphere(self, meshes: Meshes) -> Meshes:
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
        return self.vertices.shape[0]

    def __getitem__(self, idx):
        if self.steps_til_resample <= 0:
            self.set_vertices_and_normals()
            self.add_random_points()

            self.steps_til_resample = self.__len__()

        self.steps_til_resample -= 1
        return self.vertices[idx], self.normals[idx]
