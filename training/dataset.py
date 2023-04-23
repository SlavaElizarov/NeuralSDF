from abc import ABC
import os
from typing import Tuple
import numpy as np

import torch
from torch.utils.data import Dataset
import open3d as o3d
from open3d.geometry import TriangleMesh

from utils import LatinHypercubeSampler

class MeshSampler(ABC):
    def __init__(self, mesh: TriangleMesh):
        self.mesh = mesh

    def sample(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def __call__(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.sample(num_samples)


class UniformMeshSampler(MeshSampler):
    def sample(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        pointcloud = self.mesh.sample_points_uniformly(num_samples) # type: ignore
        points = np.asarray(pointcloud.points, dtype=np.float32)
        normals = np.asarray(pointcloud.normals, dtype=np.float32)
        return points, normals


class MeshDataset(Dataset):
    def __init__(
        self,
        model_path: str,
        add_vertices: bool = True,
        number_of_samples: int = 1000000,
        precompute_sdf: bool = True,
    ):
        assert os.path.exists(model_path)

        self.mesh_filepath = model_path
        self.add_vertices = add_vertices
        self.number_of_samples = number_of_samples
        self.precompute_sdf = precompute_sdf
        
        self.mesh = self._load_mesh(self.mesh_filepath)
        self.mesh_sampler = UniformMeshSampler(self.mesh)
        self.space_sampler = LatinHypercubeSampler(
            np.array([[-1.1, 1.1], [-1.1, 1.1], [-1.1, 1.1]])
        )

        self._surface_points: np.ndarray
        self._normals: np.ndarray
        self._off_surface_points: np.ndarray
        self._sdf_values: np.ndarray
        self.resample()

    def _load_mesh(self, mesh_filepath: str):
        model = o3d.io.read_triangle_model(mesh_filepath) 
        
        # next is heuristics to remove redundant geometry
        mesh = None
        for submodel in model.meshes:
            verts = np.asarray(submodel.mesh.vertices)
            # remove floating planes
            if len(verts) == 4:
                continue
            # remove vertcal 2d geometry
            if np.allclose(verts[:, 1], 0) or np.all(verts[:, 1] == verts[0, 1]):
                print(submodel.mesh_name)
                continue
            if mesh is None:
                mesh = submodel.mesh
            else:
                mesh += submodel.mesh

        return self._normalize_to_unit_sphere(mesh) # type: ignore


    def _compute_signed_distance(self, query_points: np.ndarray) -> np.ndarray:
        mesh_ = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)

        # Create a scene and add the triangle mesh
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh_)  # we do not need the geometry ID for mesh
        distances = scene.compute_signed_distance(query_points)
        return distances.numpy().astype(query_points.dtype)


    def resample(self):
        points, normals = self.mesh_sampler(self.number_of_samples)
        if self.add_vertices:
            points = np.concatenate([points, self.mesh.vertices], axis=0, dtype=points.dtype)  # type: ignore
            normals = np.concatenate([normals, self.mesh.vertex_normals], axis=0, dtype=normals.dtype)  # type: ignore

        self._surface_points = points # type: ignore
        self._normals = normals # type: ignore
        
        self._off_surface_points = self.space_sampler(
            self._surface_points.shape[0])

        if self.precompute_sdf:
            self.sdf_values = self._compute_signed_distance(self._off_surface_points)

    def _normalize_to_unit_sphere(self, mesh: TriangleMesh) -> TriangleMesh:
        v_max = np.max(mesh.vertices, axis=0)
        v_min = np.min(mesh.vertices, axis=0)
        v_center = (v_max + v_min) / 2.0
        mesh.translate(-v_center)

        v = np.asanyarray(mesh.vertices)

        # Find the max distance to origin
        max_dist = np.sqrt(np.max(np.sum(v**2, axis=-1)))
        v_scale = 1.0 / max_dist
        mesh.scale(v_scale, center=(0, 0, 0))

        return mesh

    def __len__(self):
        return max(self._surface_points.shape[0], self._off_surface_points.shape[0])

    def __getitem__(self, idx):
        if self.precompute_sdf:
            return (
                self._surface_points[idx % self._surface_points.shape[0]],
                self._normals[idx % self._normals.shape[0]],
                self._off_surface_points[
                    idx % self._off_surface_points.shape[0]],
                self.sdf_values[idx % self._off_surface_points.shape[0]]
            )
        return (
            self._surface_points[idx],
            self._normals[idx],
            self._off_surface_points[
                idx % self._off_surface_points.shape[0]
            ],  # TODO: this is a hack
        )
