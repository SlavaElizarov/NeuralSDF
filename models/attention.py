from turtle import forward
from typing import Callable, Optional, Tuple

import torch
from torch import nn
import numpy as np


class CrossAttentionLayer(nn.Module):
    def __init__(self,
                 first_input_dim: int,
                 second_input_dim: int,
                 attention_dim: int = 64,
                 number_of_heads: int = 8,
                 value_dim: Optional[int] = None,
                 use_dropout: bool = False,
                 drop_rate: float = 0.1) -> None:
        super().__init__()
        if value_dim is None:
            value_dim = attention_dim
            
        self.attention_dim = attention_dim
        self.number_of_heads = number_of_heads
        self.use_dropout = use_dropout
        self.value_dim = value_dim
        
        
        self.query_projection = nn.Linear(first_input_dim, attention_dim * number_of_heads)
        self.key_projection = nn.Linear(second_input_dim, attention_dim * number_of_heads)
        self.value_projection = nn.Linear(second_input_dim, value_dim * number_of_heads)
        
        if use_dropout:
            self.dropout = nn.Dropout(drop_rate)
        
    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        old_shape = x.shape
        feature_dim = old_shape[-1] // self.number_of_heads
        new_shape = old_shape[:-1] + (self.number_of_heads, feature_dim)
        return x.view(*new_shape)
        
    def forward(self, x, y) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        query = self.query_projection(x)
        query = self.separate_heads(query)
        key = self.separate_heads(self.key_projection(y))
        value = self.separate_heads(self.value_projection(y))
        
        scores = torch.einsum('b n h d, b m h d -> b h n m', query, key) # (batch_size, num_heads, query_length, key_length)
        scores = scores / np.sqrt(self.attention_dim)
        
        attention = torch.softmax(scores, dim=-1)
        
        if self.use_dropout:
            self.dropout(attention)

        out = torch.einsum('b h n m, b m h d -> b n h d', attention, value) # (batch_size, query_length, num_heads, value_dim)
        
        return out, attention
        
        
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

        # TODO: Do ypu need to project keys? Looks pretty fixed
        self.keys_projections = torch.nn.ModuleList([nn.Linear(input_dim, attention_dim) for _ in range(n_heads)])
        self.values_projections = torch.nn.ModuleList(
            [values_projection_factory(input_dim, output_dim, i) for i in range(n_heads)]
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

        return torch.einsum("bh,boh->bo", attention, values), attention  # (batch_size, output_dim)
    
class ImplicitAttetionLayerLite(nn.Module):
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
        
        self.keys = nn.Linear(attention_dim, n_heads, bias=False)
        torch.nn.init.orthogonal_(self.keys.weight)
        
        self.values_projections = torch.nn.ModuleList(
            [values_projection_factory(input_dim, output_dim, i) for i in range(n_heads)]
        )
        self.query_projection = torch.nn.Linear(input_dim, attention_dim)
        self.scale_dot = scale_dot

    def forward(self, x):
        query = self.query_projection(x)  # (batch_size, attention_dim)

        values = torch.stack(
            [value_projection(x) for value_projection in self.values_projections], dim=-1
        )  # (batch_size, output_dim, n_heads)

        attention_dot = self.keys(query)  # (batch_size, n_heads)
        if self.scale_dot:
            attention_dot = attention_dot / np.sqrt(self.attention_dim)  # Following the paper Attention is all you need

        attention = torch.softmax(attention_dot, dim=-1)  # (batch_size, n_heads)

        return torch.einsum("bh,boh->bo", attention, values), attention  # (batch_size, output_dim)
    