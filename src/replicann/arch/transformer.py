"""
Transformer blocks implemented after https://arxiv.org/abs/1706.03762.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from replicann.nn.attention import MultiheadCrossAttention, MultiheadSelfAttention


class _TransformerFFN(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        *,
        bias: bool = True,
        hidden_size: int | None = None,
        p_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_size = hidden_size or 4 * embedding_size
        self._upscale = nn.Linear(embedding_size, hidden_size, bias=bias)
        self._downscale = nn.Linear(hidden_size, embedding_size, bias=bias)
        self._dropout = nn.Dropout(p_dropout)

    def forward(self, x: Tensor, /) -> None:
        x = self._upscale(x)
        x = F.relu(x)
        x = self._downscale(x)
        x = self._dropout(x)
        return x

    @property
    def embedding_size(self) -> int:
        return self._upscale.in_features
    
    @property
    def hidden_size(self) -> int:
        return self._upscale.out_features


class _TransformerBlock(nn.Module):
    def __init__(
        self,
        n_heads: int,
        embedding_size: int,
        *,
        ffn_bias: bool = True,
        ffn_hidden_size: int | None = None,
        head_bias: bool = False,
        proj_bias: bool = True,
        p_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._attn = MultiheadSelfAttention(
            n_heads,
            embedding_size // n_heads,
            embedding_size,
            head_bias=head_bias,
            proj_bias=proj_bias,
            p_dropout=p_dropout,
        )
        self._attn_ln = nn.LayerNorm(embedding_size)
        self._ffn = _TransformerFFN(
            embedding_size,
            bias=ffn_bias,
            hidden_size=ffn_hidden_size,
            p_dropout=p_dropout,
        )
        self._ffn_ln = nn.LayerNorm(embedding_size)

    @property
    def embedding_size(self) -> int:
        return self._attn.embedding_size
    
    @property
    def head_size(self) -> int:
        return self._attn.head_size
    
    @property
    def n_heads(self) -> int:
        return self._attn.n_heads
    

class TransformerEncoder(_TransformerBlock):
    def __init__(
        self,
        n_heads: int,
        embedding_size: int,
        *,
        ffn_bias: bool = True,
        ffn_hidden_size: int | None = None,
        head_bias: bool = False,
        proj_bias: bool = True,
        p_dropout: float = 0.1, 
    ) -> None:
        super().__init__(
            n_heads=n_heads,
            embedding_size=embedding_size,
            ffn_bias=ffn_bias,
            ffn_hidden_size=ffn_hidden_size,
            head_bias=head_bias,
            proj_bias=proj_bias,
            p_dropout=p_dropout, 
        )

    def forward(self, x: Tensor, *, return_kv: bool = False) -> Tensor:
        if not return_kv:
            x = self._attn_ln(x + self._attn(x))
            x = self._ffn_ln(x + self._ffn(x))
            return x
        z, k, v = self._attn(x, return_kv=True)
        x = self._attn_ln(x + z)
        x = self._ffn_ln(x + self._ffn(x))
        return x, k, v
    

class TransformerDecoder(_TransformerBlock):
    def __init__(
        self,
        n_heads: int,
        embedding_size: int,
        *,
        context_size: int,
        ffn_bias: bool = True,
        ffn_hidden_size: int | None = None,
        head_bias: bool = False,
        proj_bias: bool = True,
        p_dropout: float = 0.1, 
    ) -> None:
        super().__init__(
            n_heads=n_heads,
            embedding_size=embedding_size,
            ffn_bias=ffn_bias,
            ffn_hidden_size=ffn_hidden_size,
            head_bias=head_bias,
            proj_bias=proj_bias,
            p_dropout=p_dropout, 
        )
        self.register_buffer(
            "_attn_mask",
            torch.tril(torch.ones(context_size, context_size)),
        )

    def forward(self, x: Tensor) -> Tensor:
        n_tokens = x.shape[1]
        x = self._attn_ln(x + self._attn(x, mask=self._attn_mask[:n_tokens, :n_tokens]))
        x = self._ffn_ln(x + self._ffn(x))
        return x
    

class TransformerCrossDecoder(_TransformerBlock):
    def __init__(
        self,
        n_heads: int,
        embedding_size: int,
        *,
        context_size: int,
        ffn_bias: bool = True,
        ffn_hidden_size: int | None = None,
        head_bias: bool = False,
        proj_bias: bool = True,
        p_dropout: float = 0.1, 
    ) -> None:
        super().__init__(
            n_heads=n_heads,
            embedding_size=embedding_size,
            ffn_bias=ffn_bias,
            ffn_hidden_size=ffn_hidden_size,
            head_bias=head_bias,
            proj_bias=proj_bias,
            p_dropout=p_dropout,
        )
        self.register_buffer(
            "_attn_mask",
            torch.tril(torch.ones(context_size, context_size)),
        )
        self._cross_attn = MultiheadCrossAttention(
            n_heads,
            embedding_size // n_heads,
            embedding_size,
            head_bias=head_bias,
            proj_bias=proj_bias,
            p_dropout=p_dropout,
        )
        self._cross_attn_ln = nn.LayerNorm(embedding_size)

    def forward(
        self,
        x: Tensor,
        /,
        encoder_k: Tensor,
        encoder_v: Tensor,
    ) -> Tensor:
        n_tokens = x.shape[1]
        x = self._attn_ln(x + self._attn(x, mask=self._attn_mask[:n_tokens, :n_tokens]))
        x = self._cross_attn_ln(x + self._cross_attn(x, encoder_k, encoder_v))
        x = self._ffn_ln(x + self._ffn(x))
        return x
