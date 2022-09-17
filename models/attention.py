from typing import Callable

import torch
from torch import nn
import numpy as np


class ImplicitAttetionLayer(nn.Module):
    def __init__(
        self,
        n_heads: int = 1,
        input_dim: int = 256,
        attention_dim: int = 64,
        output_dim: int = 256,
        values_projection_factory: Callable[[int, int, int], nn.Module] = lambda input_dim, output_dim, head_id: nn.Linear(
            input_dim, output_dim
        ),
        scale_dot: bool = True,
    ):
        """
        Implicit Attention Layer
        It resembles Multi-Head Attention Layer, but attention is choosing between heads

        K_h = fk_h(X) - key transform, h-th head
        Q = fq(X) - query transform
        V_h = fv_h(X) - value transform, h-th head

        y = sum_h(softmax(Q * K) * V)


        Args:
            n_heads (int, optional): Number of heads. Defaults to 1.
            input_dim (int, optional): Input dimention. Defaults to 256.
            attention_dim (int, optional): Size of Q and K_h. Defaults to 64.
            output_dim (int, optional): Size of V_h. Defaults to 256.
            values_projection_factory (Callable[[int, int], nn.Module], optional):
                    Factory function for creating input to values projection. Defaults to Linear projection.
        """
        super().__init__()
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.attention_dim = attention_dim

        self.keys_projections = torch.nn.ModuleList([nn.Linear(input_dim, attention_dim) for _ in range(n_heads)])
        self.values_projections = torch.nn.ModuleList(
            [values_projection_factory(input_dim, output_dim) for _ in range(n_heads)]
        )
        self.query_projection = torch.nn.Linear(input_dim, attention_dim)
        self.scale_dot = scale_dot

    def forward(self, x):
        query = self.query_projection(x)  # (batch_size, attention_dim)
        keys = torch.stack(
            [key_projection(x) for key_projection in self.keys_projections], dim=-1
        )  # (batch_size, attention_dim, n_heads)
        values = torch.stack(
            [value_projection(x) for value_projection in self.values_projections], dim=-1
        )  # (batch_size, output_dim, n_heads)

        attention_dot = torch.einsum("ba,bah->bh", query, keys)  # (batch_size, n_heads)
        if self.scale_dot:
            attention_dot = attention_dot / np.sqrt(self.attention_dim)  # Following the paper Attention is all you need

        attention = torch.softmax(attention_dot, dim=-1)  # (batch_size, n_heads)

        return torch.einsum("bh,boh->bo", attention, values)  # (batch_size, output_dim)
