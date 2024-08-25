"""
Scaled dot-product self-attention and cross-attention heads and their multihead
variants implemented after https://arxiv.org/abs/1706.03762.
"""

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _AttentionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        bias: bool = False,
        p_dropout: float = 0.1
    ) -> None:
        super().__init__()
        self._scale_coeff = 1 / sqrt(in_dim)
        self._query = nn.Linear(in_dim, out_dim, bias=bias)
        self._dropout = nn.Dropout(p_dropout)

    def _attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor | None = None,
        *,
        return_kv: bool = False,
    ) -> Tensor:
        w = q @ k.transpose(-2, -1) * self._scale_coeff
        if mask is not None:
            if (mdtype := mask.dtype) == torch.bool:
                w = torch.masked_fill(w, mask, float("-inf"))
            elif mdtype == torch.float:
                w += mask
            else:
                raise TypeError(f"Unexpected type `{mdtype}` of the `mask`.")
        w = F.softmax(w, dim=-1)
        w = self._dropout(w)
        z = w @ v
        if return_kv:
            return z, k, v
        return z

    @property
    def embeddings_size(self) -> int:
        return self.query.in_features

    @property
    def head_size(self) -> int:
        return self.query.out_features

    @property
    def query(self) -> nn.Linear:
        return self._query
    

class _MultiheadAttention(nn.Module):
    AttentionHead: type

    def __init__(
        self,
        n_heads: int,
        head_size: int,
        embedding_size: int | None = None,
        *,
        head_bias: bool = False,
        proj_bias: bool = True,
        p_dropout: float = 0.1,
        **head_kwargs,
    ) -> None:
        super().__init__()
        embedding_size = embedding_size or head_size
        self._heads = nn.ModuleList(
            self.AttentionHead(embedding_size, head_size, bias=head_bias, **head_kwargs)
            for _ in range(n_heads)
        )
        self._proj = nn.Linear(n_heads * head_size, embedding_size, bias=proj_bias)
        self._dropout = nn.Dropout(p_dropout)

    @property
    def embedding_size(self) -> int:
        return self._proj.out_features
    
    @property
    def head_size(self) -> int:
        return self._proj.in_features // self.n_heads

    @property
    def n_heads(self) -> int:
        return len(self._heads)


class CrossAttentionHead(_AttentionHead):
    def forward(
        self,
        x: Tensor,
        /,
        k: Tensor,
        v: Tensor,
        mask: Tensor | None = None,
        *,
        return_kv: bool = False,
    ) -> Tensor:
        q = self.query(x)
        return self._attention(q, k, v, mask=mask, return_kv=return_kv)
    

class SelfAttentionHead(_AttentionHead):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        bias: bool = False,
        p_dropout: float = 0.1
    ) -> None:
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            bias=bias,
            p_dropout=p_dropout,
        )
        self._key = nn.Linear(in_dim, out_dim, bias=bias)
        self._value = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: Tensor, /, mask: Tensor | None = None, return_kv: bool = False) -> Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        return self._attention(q, k, v, mask=mask, return_kv=return_kv)

    @property
    def key(self) -> nn.Linear:
        return self._key

    @property
    def value(self) -> nn.Linear:
        return self._value
    

class MultiheadCrossAttention(_MultiheadAttention):
    AttentionHead = CrossAttentionHead

    def forward(self, x: Tensor, /, k: Tensor, v: Tensor, mask: Tensor | None = None, *, return_kv: bool = False) -> Tensor:
        k_parts = torch.split(k, (split_size := self.embedding_size // self.n_heads), dim=-1)[:self.n_heads]
        v_parts = torch.split(v, split_size, dim=-1)[:self.n_heads]
        heads_with_kv = zip(self._heads, k_parts, v_parts)
        if not return_kv:
            y = torch.cat([
                h(x, kp, vp, mask=mask)
                for h, kp, vp in heads_with_kv
            ], dim=-1)
            y = self._dropout(self._proj(y))
            return y
        zs, ks, vs = [], [], []
        for h, kp, vp in heads_with_kv:
            z, k_, v_ = h(x, kp, vp, mask=mask)
            zs.append(z)
            ks.append(k_)
            vs.append(v_)
        zs = torch.cat(zs, dim=-1)
        ks = torch.cat(ks, dim=-1)
        vs = torch.cat(vs, dim=-1)
        return zs, ks, vs


class MultiheadSelfAttention(_MultiheadAttention):
    AttentionHead = SelfAttentionHead

    def forward(self, x: Tensor, /, mask: Tensor | None = None, *, return_kv: bool = False) -> Tensor:
        if not return_kv:
            y = torch.cat([h(x, mask=mask) for h in self._heads], dim=-1)
            y = self._dropout(self._proj(y))
            return y
        zs, ks, vs = [], [], []
        for h in self._heads:
            z, k, v = h(x, mask=mask, return_kv=return_kv)
            zs.append(z)
            ks.append(k)
            vs.append(v)
        zs = torch.cat(zs, dim=-1)
        ks = torch.cat(ks, dim=-1)
        vs = torch.cat(vs, dim=-1)
        return zs, ks, vs
