import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple
from torch.utils.checkpoint import checkpoint

import sys, os
_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(_dir, '..', 'Attention'))
sys.path.append(os.path.join(_dir, '..', 'FeedForward'))
sys.path.append(os.path.join(_dir, '..', 'TransformerBlock'))

from gdn_attention import RMSNorm, GDNState
from gdn_block import GDNBlock
from attn_block import NaylisAttnBlock
from attention import KVCache


class GNT(nn.Module):
    def __init__(
        self,
        vocab_size                : int   = 49_152,
        embed_dim                 : int   = 768,
        num_heads                 : int   = 12,
        n_kv_heads                : int   = 4,
        num_layers                : int   = 32,
        rel_rank                  : int   = 48,
        max_seq_len               : int   = 2048,
        dropout                   : float = 0.0,
        attn_every_n_layers       : int   = 4,
        use_qk_norm               : bool  = True,
        conv_kernel               : int   = 4,
        use_yarn                  : bool  = False,
        yarn_scale                : float = 1.0,
        yarn_original_max_len     : int   = 2048,
        use_gradient_checkpointing: bool  = False,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0
        assert num_heads % n_kv_heads == 0

        self.vocab_size          = vocab_size
        self.embed_dim           = embed_dim
        self.num_heads           = num_heads
        self.n_kv_heads          = n_kv_heads
        self.num_layers          = num_layers
        self.rel_rank            = rel_rank
        self.max_seq_len         = max_seq_len
        self.attn_every_n_layers = attn_every_n_layers
        self.use_grad_ckpt       = use_gradient_checkpointing

        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.dropout          = nn.Dropout(dropout)

        self.blocks      = nn.ModuleList()
        self.block_types = []

        for i in range(num_layers):
            is_attn = ((i + 1) % attn_every_n_layers == 0)
            self.block_types.append('attn' if is_attn else 'gdn')

            if is_attn:
                self.blocks.append(NaylisAttnBlock(
                    embed_dim             = embed_dim,
                    num_heads             = num_heads,
                    dropout               = dropout,
                    use_rope              = True,
                    max_seq_len           = max_seq_len,
                    use_yarn              = use_yarn,
                    yarn_scale            = yarn_scale,
                    yarn_original_max_len = yarn_original_max_len,
                    use_swiglu            = True,
                    n_kv_heads            = n_kv_heads,
                    use_qk_norm           = use_qk_norm,
                    use_flash_attn        = True,
                    rel_rank              = rel_rank,
                ))
            else:
                self.blocks.append(GDNBlock(
                    embed_dim   = embed_dim,
                    num_heads   = num_heads,
                    n_kv_heads  = n_kv_heads,
                    rel_rank    = rel_rank,
                    dropout     = dropout,
                    use_qk_norm = use_qk_norm,
                    conv_kernel = conv_kernel,
                ))

        n_gdn  = self.block_types.count('gdn')
        n_attn = self.block_types.count('attn')
        print(f'  GNT blocks : {n_gdn} GDN + {n_attn} Attn = {num_layers} total')
        if use_gradient_checkpointing:
            print('  Gradient checkpointing : ON')

        self.ln_final    = RMSNorm(embed_dim)
        self.output_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.output_head.weight = self.token_embeddings.weight

        self.apply(self._init_weights)

        for block in self.blocks:
            attn = getattr(block, 'attention', None)
            if attn is None:
                continue
            if hasattr(attn, 'forget_gate'):
                nn.init.constant_(attn.forget_gate.bias, 2.0)
            if hasattr(attn, 'rel_q_proj'):
                nn.init.normal_(attn.rel_q_proj.weight, std=0.01)
                nn.init.normal_(attn.rel_k_proj.weight, std=0.01)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def _gdn_block_forward(self, block, x):
        out, _ = block(x, state=None, use_cache=False)
        return out

    def _attn_block_forward(self, block, x):
        out, _ = block(x, past_kv=None, use_kv_cache=False)
        return out

    def forward(
        self,
        input_ids   : torch.Tensor,
        targets     : Optional[torch.Tensor] = None,
        past_states : Optional[List]         = None,
        use_cache   : bool                   = False,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int]          = None,
        max_seqlen_k: Optional[int]          = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List]]:

        x = self.dropout(self.token_embeddings(input_ids))

        if x.device.type == 'cuda' and x.dtype == torch.float32:
            x = x.to(torch.bfloat16)

        use_ckpt   = self.use_grad_ckpt and self.training and not use_cache
        new_states = [] if use_cache else None

        for i, (block, btype) in enumerate(zip(self.blocks, self.block_types)):
            past = past_states[i] if past_states is not None else None

            if use_ckpt:
                if btype == 'gdn':
                    x = checkpoint(self._gdn_block_forward, block, x, use_reentrant=False)
                else:
                    x = checkpoint(self._attn_block_forward, block, x, use_reentrant=False)
            else:
                if btype == 'gdn':
                    x, new_s = block(x, state=past, use_cache=use_cache)
                else:
                    x, new_s = block(
                        x,
                        past_kv      = past,
                        use_kv_cache = use_cache,
                        cu_seqlens_q = cu_seqlens_q,
                        cu_seqlens_k = cu_seqlens_k,
                        max_seqlen_q = max_seqlen_q,
                        max_seqlen_k = max_seqlen_k,
                    )
                if use_cache:
                    new_states.append(new_s)

        x      = self.ln_final(x)
        logits = self.output_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss, new_states

    def generate(
        self,
        input_ids     : torch.Tensor,
        max_new_tokens: int            = 100,
        temperature   : float          = 1.0,
        top_k         : Optional[int]  = None,
        top_p         : Optional[float]= None,
        eos_token_id  : Optional[int]  = None,
        stop_token_ids: Optional[List[int]] = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()

        with torch.no_grad():
            _, _, past_states = self.forward(input_ids, use_cache=True)
            cur_ids = input_ids

            for _ in range(max_new_tokens):
                logits, _, past_states = self.forward(
                    cur_ids[:, -1:], past_states=past_states, use_cache=True
                )
                next_logits = logits[:, -1, :]

                if temperature == 0.0:
                    next_tok = next_logits.argmax(dim=-1, keepdim=True)
                else:
                    next_logits = next_logits / temperature

                    if top_k is not None:
                        k_      = min(top_k, next_logits.size(-1))
                        v, _    = torch.topk(next_logits, k_)
                        next_logits = next_logits.masked_fill(
                            next_logits < v[:, [-1]], float('-inf')
                        )

                    if top_p is not None and top_p < 1.0:
                        sorted_l, sorted_i = torch.sort(next_logits, descending=True)
                        cum_probs          = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
                        sorted_l           = sorted_l.masked_fill(cum_probs > top_p, float('-inf'))
                        sorted_l[..., 0]   = sorted_l[..., 0].clamp(min=0)
                        next_logits        = torch.full_like(next_logits, float('-inf')).scatter_(
                            1, sorted_i, sorted_l
                        )

                    next_tok = torch.multinomial(F.softmax(next_logits, dim=-1), 1)

                cur_ids = torch.cat([cur_ids, next_tok], dim=1)

                # Construire l'ensemble des tokens d'arrêt (EOS + stop_token_ids)
                all_stop_ids = set()
                if eos_token_id is not None:
                    all_stop_ids.add(eos_token_id)
                if stop_token_ids:
                    all_stop_ids.update(stop_token_ids)
                if all_stop_ids and next_tok.item() in all_stop_ids:
                    break

        if was_training:
            self.train()
        return cur_ids

    def count_parameters(self) -> dict:
        total  = sum(p.numel() for p in self.parameters())
        naylis = sum(
            p.numel()
            for b in self.blocks
            for name, p in b.named_parameters()
            if any(k in name for k in ('rel_q_proj', 'rel_k_proj', 'graph_scale'))
        )
        gdn_params = sum(
            p.numel()
            for i, b in enumerate(self.blocks)
            if self.block_types[i] == 'gdn'
            for p in b.parameters()
        )
        return {
            'total_M'    : round(total      / 1e6, 2),
            'naylis_K'   : round(naylis     / 1e3, 1),
            'gdn_M'      : round(gdn_params / 1e6, 2),
            'naylis_pct' : f'{naylis / total * 100:.2f}%',
        }

    def resize_token_embeddings(self, new_vocab_size: int):
        if new_vocab_size == self.vocab_size:
            return
        old = self.token_embeddings
        self.token_embeddings = nn.Embedding(new_vocab_size, self.embed_dim)
        n = min(old.num_embeddings, new_vocab_size)
        with torch.no_grad():
            self.token_embeddings.weight.data[:n] = old.weight.data[:n]
        self.output_head        = nn.Linear(self.embed_dim, new_vocab_size, bias=False)
        self.output_head.weight = self.token_embeddings.weight
        self.vocab_size         = new_vocab_size