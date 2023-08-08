import numpy as np
import torch


class LatinHypercubeSampler:
    def __init__(self, region: np.ndarray):
        """
        Generate Latin hypercube samples within an arbitrary region.

        More information about Latin hypercube sampling can be found here:
        https://en.wikipedia.org/wiki/Latin_hypercube_sampling
        https://projecteuclid.org/journals/annals-of-statistics/volume-24/issue-5/On-Latin-hypercube-sampling/10.1214/aos/1069362310.pdf


        Args:
            region (np.ndarray): The region to sample from.
                              Format: [[min_val_1, max_val_1], ..., [min_val_d, max_val_d]]
                              where d is the dimension of the region.
        """
        assert len(region) > 0, "Region must have at least one dimension."
        assert (
            region.shape[1] == 2
        ), "Region must have two columns (min and max values)."
        assert np.all(
            region[:, 0] < region[:, 1]
        ), "Min values must be less than max values."
        self.region = region
        self.dim = region.shape[0]

    def __call__(self, n_samples: int) -> np.ndarray:
        return self.sample(n_samples)

    def sample(self, n_samples: int) -> np.ndarray:
        assert n_samples > 0, "Number of samples must be greater than 0."

        points_per_dim = np.floor(np.power(n_samples, 1 / self.dim)).astype(int)
        n_samples_int = int(np.power(points_per_dim, self.dim))
        if n_samples_int != n_samples:
            # warnings.warn(f"Number of samples ({n_samples}) is not a power of {points_per_dim}. Using {n_samples_int} samples instead.")
            n_samples = n_samples_int
        # calculate cell size
        cell_size = (self.region[:, 1] - self.region[:, 0]) / points_per_dim
        
        # generate random samples wrt the center of each cell
        points = np.random.uniform(
            cell_size * -0.5,
            cell_size * 0.5,
            size= (n_samples, len(cell_size))
        ).astype(np.float32)

        sides = []
        for side in self.region:
            sides.append(
                np.linspace(side[0], side[1], points_per_dim, dtype=np.float32)
            )
        grid = np.stack(np.meshgrid(*sides, indexing="ij"), axis=-1)
        grid = grid.reshape(n_samples, self.dim)
        grid += cell_size * 0.5  # shift grid to the center of the cell
        grid += points
        return grid