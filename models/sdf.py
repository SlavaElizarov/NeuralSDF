
from abc import ABC, abstractmethod
import torch
from torch import nn

class SDF(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass