import torch
import torch.nn as nn
from typing import Optional, Tuple

from gdn_attention import GDNNaylisAttention, RMSNorm, GDNState
from feedforward import FeedForward


class GDNBlock(nn.Module):

    def __init__(
        self,
        embed_dim   : int,
        num_heads   : int,
        n_kv_heads  : Optional[int] = None,
        rel_rank    : int   = 48,
        dropout     : float = 0.0,
        use_qk_norm : bool  = True,
        conv_kernel : int   = 4,
    ):
        super().__init__()
        self.ln1 = RMSNorm(embed_dim)
        self.attention = GDNNaylisAttention(
            embed_dim   = embed_dim,
            num_heads   = num_heads,
            n_kv_heads  = n_kv_heads,
            rel_rank    = rel_rank,
            dropout     = dropout,
            use_qk_norm = use_qk_norm,
            conv_kernel = conv_kernel,
        )
        self.ln2 = RMSNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, dropout, use_swiglu=True)

    def forward(
        self,
        x        : torch.Tensor,
        state    : Optional[GDNState] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[GDNState]]:

        attn_out, new_state = self.attention(
            self.ln1(x),
            state     = state,
            use_cache = use_cache,
        )
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_state