from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn
import numpy as np


class CrossAttentionLayer(nn.Module):
    def __init__(
        self,
        first_in_features: int,
        second_in_features: int,
        attention_dim: int = 64,
        number_of_heads: int = 8,
        value_dim: Optional[int] = None,
        use_dropout: bool = False,
        drop_rate: float = 0.1,
    ) -> None:
        """
        Cross Attention Layer with multiple heads.
        Implementation based on https://arxiv.org/abs/1706.03762

        Cross attention is used to compute attention between two sequences of tokens.
        It can be used to compute self-attention for a single sequence of tokens.

        Attention is computed as follows:
        Q = W_q * x
        K = W_k * y
        V = W_v * y
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

        Where W_q, W_k, W_v stand for query, key and value projections respectively.
        x is the query sequence, y is the key-value sequence.

        Args:
            first_in_features (int): Feature dimension of the first input sequence.
            second_in_features (int): Feature dimension of the second input sequence.
            attention_dim (int, optional): Dim of queries and keys. Defaults to 64.
            number_of_heads (int, optional): Number of heads. Defaults to 8.
            value_dim (Optional[int], optional): Feature dim of a value tokens and output. Defaults to None.
            use_dropout (bool, optional): Apply dropout to attention weights (will nullify random tokens). Defaults to False.
            drop_rate (float, optional): Dropout rate. Defaults to 0.1.
        """
        super().__init__()
        if value_dim is None:
            value_dim = attention_dim

        self.attention_dim = attention_dim
        self.number_of_heads = number_of_heads
        self.use_dropout = use_dropout
        self.value_dim = value_dim

        self.query_projection = nn.Linear(
            first_in_features, attention_dim * number_of_heads
        )
        self.key_projection = nn.Linear(
            second_in_features, attention_dim * number_of_heads
        )
        self.value_projection = nn.Linear(
            second_in_features, value_dim * number_of_heads
        )

        if use_dropout:
            self.dropout = nn.Dropout(drop_rate)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        old_shape = x.shape
        feature_dim = old_shape[-1] // self.number_of_heads
        new_shape = old_shape[:-1] + (self.number_of_heads, feature_dim)
        return x.view(*new_shape)

    def forward(
        self,
        first_seq: torch.Tensor,
        second_seq: torch.Tensor,
        return_scores: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute cross attention between two sequences of tokens.
        First sequence is the query sequence, second sequence is the key-value sequence.

        Args:
            first_seq (torch.Tensor): Query sequence of shape (batch_size, q_len, first_in_features).
            second_seq (torch.Tensor): Key-value sequence of shape (batch_size, k_len, second_in_features).
            return_scores (bool, optional): Return attention scores. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Output of shape (batch_size, q_len, value_dim) and attention scores of shape (batch_size, number_of_heads, q_len, k_len).
        """
        if len(first_seq.shape) == 2:
            first_seq = first_seq.unsqueeze(1)

        query = self.query_projection(first_seq)
        query = self.separate_heads(query)
        key = self.separate_heads(self.key_projection(second_seq))
        value = self.separate_heads(self.value_projection(second_seq))

        scores = torch.einsum(
            "b n h d, b m h d -> b h n m", query, key
        )  # (batch_size, num_heads, query_length, key_length)
        scores = scores / np.sqrt(self.attention_dim)

        attention = torch.softmax(scores, dim=-1)

        if self.use_dropout:
            self.dropout(attention)

        out = torch.einsum(
            "b h n m, b m h d -> b n h d", attention, value
        )  # (batch_size, query_length, num_heads, value_dim)

        if return_scores:
            return out, attention
        return out


class SubtractionCrossAttentionLayer(CrossAttentionLayer):
    def __init__(
        self,
        first_in_features: int,
        second_in_features: int,
        attention_dim: int = 64,
        number_of_heads: int = 8,
        value_dim: Optional[int] = None,
        use_dropout: bool = False,
        drop_rate: float = 0.1,
    ) -> None:
        """
        Subtraction-based Cross Attention Layer
        proposed in the paper "Point Transformer" https://arxiv.org/abs/2012.09164
        and empirically shown to be more efficient than the original Cross Attention Layer
        in "IS ATTENTION ALL THAT NERF NEEDS?" https://arxiv.org/abs/2207.13298
        Official implementation: https://github.com/VITA-Group/GNT/blob/main/gnt/transformer_network.py#L55

        Attention is computed as follows:
        Attention(Q, K, V) = softmax(W_p * (Q - K) / sqrt(d_k)) V

        CAUTION: This layer is memory-intensive and should be used with care.

        Args:
            first_in_features (int): Feature dimension of the first input sequence.
            second_in_features (int): Feature dimension of the second input sequence.
            attention_dim (int, optional): Dim of queries and keys. Defaults to 64.
            number_of_heads (int, optional): Number of heads. Defaults to 8.
            value_dim (Optional[int], optional): Feature dim of a value tokens and output. Defaults to None.
            use_dropout (bool, optional): Apply dropout to attention weights (will nullify random tokens). Defaults to False.
            drop_rate (float, optional): Dropout rate. Defaults to 0.1.
        """
        super().__init__(
            first_in_features,
            second_in_features,
            attention_dim,
            number_of_heads,
            value_dim,
            use_dropout,
            drop_rate,
        )

        self.score_projection = nn.Linear(attention_dim, self.value_dim)

    def forward(
        self,
        first_seq: torch.Tensor,
        second_seq: torch.Tensor,
        return_scores: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute subtraction-based cross attention between two sequences of tokens.
        First sequence is the query sequence, second sequence is the key-value sequence.

        Args:
            first_seq (torch.Tensor): Query sequence of shape (batch_size, q_len, first_in_features).
            second_seq (torch.Tensor): Key-value sequence of shape (batch_size, k_len, second_in_features).
            return_scores (bool, optional): Return attention scores. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Output of shape (batch_size, q_len, value_dim) and attention scores of shape (batch_size, number_of_heads, q_len, k_len).
        """
        if len(first_seq.shape) == 2:
            first_seq = first_seq.unsqueeze(1)

        query = self.separate_heads(
            self.query_projection(first_seq)
        )  # (batch_size, query_length, num_heads, attention_dim)
        key = self.separate_heads(
            self.key_projection(second_seq)
        )  # (batch_size, key_length, num_heads, attention_dim)
        value = self.separate_heads(
            self.value_projection(second_seq)
        )  # (batch_size, key_length, num_heads, value_dim)

        raw_scores = key.unsqueeze_(1) - query.unsqueeze_(2)
        scores = self.score_projection(
            raw_scores
        )  # (batch_size, query_length, key_length, num_heads, value_dim)
        scores = scores / np.sqrt(self.attention_dim)
        attention = torch.softmax(scores, dim=2)

        if self.use_dropout:
            self.dropout(attention)

        out = torch.einsum(
            "b n m h d, b m h d -> b n h d", attention, value
        )  # (batch_size, query_length, num_heads, value_dim)

        if return_scores:
            return out, attention
        return out


class CommutatorAttetionLayer(nn.Module):
    def __init__(
        self,
        n_heads: int = 1,
        in_features: int = 256,
        attention_dim: int = 64,
        out_features: int = 256,
        values_projection_factory: Callable[
            [int, int, int], nn.Module
        ] = lambda in_features, out_features, head_id: nn.Linear(
            in_features, out_features
        ),
        scale_dot: bool = True,
    ):
        """
        Artsy-fartsy attention layer, building block of the Commutator Siren.
        It resembles Multi-Head Attention Layer, but attention is weighting heads instead of values

        K_h = fk_h(X) - key transform, h-th head
        Q = fq(X) - query transform
        V_h = fv_h(X) - value transform, h-th head

        y = sum_h(softmax(Q * K) * V)


        Args:
            n_heads (int, optional): Number of heads. Defaults to 1.
            in_features (int, optional): Input dimention. Defaults to 256.
            attention_dim (int, optional): Size of Q and K_h. Defaults to 64.
            out_features (int, optional): Size of V_h. Defaults to 256.
            values_projection_factory (Callable[[int, int], nn.Module], optional):
                    Factory function for creating input to values projection. Defaults to Linear projection.
        """
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.attention_dim = attention_dim

        # TODO: Do ypu need to project keys? Looks pretty fixed
        self.keys_projections = torch.nn.ModuleList(
            [nn.Linear(in_features, attention_dim) for _ in range(n_heads)]
        )
        self.values_projections = torch.nn.ModuleList(
            [
                values_projection_factory(in_features, out_features, i)
                for i in range(n_heads)
            ]
        )
        self.query_projection = torch.nn.Linear(in_features, attention_dim)
        self.scale_dot = scale_dot

    def forward(self, x):
        query = self.query_projection(x)  # (batch_size, attention_dim)
        keys = torch.stack(
            [key_projection(x) for key_projection in self.keys_projections], dim=-1
        )  # (batch_size, attention_dim, n_heads)

        values = torch.stack(
            [value_projection(x) for value_projection in self.values_projections],
            dim=-1,
        )  # (batch_size, out_features, n_heads)

        attention_dot = torch.einsum("ba,bah->bh", query, keys)  # (batch_size, n_heads)
        if self.scale_dot:
            attention_dot = attention_dot / np.sqrt(
                self.attention_dim
            )  # Following the paper Attention is all you need

        attention = torch.softmax(attention_dot, dim=-1)  # (batch_size, n_heads)

        return (
            torch.einsum("bh,boh->bo", attention, values),
            attention,
        )  # (batch_size, out_features)
