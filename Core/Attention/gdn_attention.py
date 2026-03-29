import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

_HAS_FLA_GATED = False
_HAS_FLA_DELTA = False

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _fla_gated_rule
    _HAS_FLA_GATED = True
    print('  ⚡ FLA chunk_gated_delta_rule : OK')
except ImportError:
    print('  ⚠️  FLA chunk_gated_delta_rule non disponible')

if not _HAS_FLA_GATED:
    try:
        from fla.ops.delta_rule import chunk_delta_rule as _fla_delta_rule
        _HAS_FLA_DELTA = True
        print('  ⚠️  FLA chunk_delta_rule disponible — alpha encodé approximativement')
    except ImportError:
        print('  ⚠️  FLA non disponible — fallback Python')

FORCE_PYTHON_SCAN = not _HAS_FLA_GATED


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class ShortConv1d(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(
            dim, dim,
            kernel_size = kernel_size,
            padding     = kernel_size - 1,
            groups      = dim,
            bias        = False,
        )
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor, conv_cache: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape
        if conv_cache is not None and S == 1:
            x_padded = torch.cat([conv_cache, x], dim=1)
        else:
            x_padded = x
        new_cache = x_padded[:, -(self.kernel_size - 1):, :].detach()
        x_t       = x_padded.transpose(1, 2)
        out       = self.conv(x_t)
        out       = out[:, :, :x_padded.shape[1]]
        out       = out[:, :, -S:]
        return out.transpose(1, 2), new_cache


class GDNState:
    def __init__(self, S: torch.Tensor, conv_caches: Optional[list] = None):
        self.S           = S
        self.conv_caches = conv_caches

    def clone(self) -> 'GDNState':
        caches = [c.clone() for c in self.conv_caches] if self.conv_caches is not None else None
        return GDNState(self.S.clone(), caches)


def _python_chunk_scan(
    k_tilde   : torch.Tensor,
    v         : torch.Tensor,
    q_tilde   : torch.Tensor,
    beta      : torch.Tensor,
    alpha     : torch.Tensor,
    mem_init  : torch.Tensor,
    chunk_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, H, S, d = k_tilde.shape
    mem        = mem_init.clone()
    out_chunks = []

    for start in range(0, S, chunk_size):
        end  = min(start + chunk_size, S)
        T    = end - start
        kt   = k_tilde[:, :, start:end, :]
        vt   = v      [:, :, start:end, :]
        qt   = q_tilde[:, :, start:end, :]
        at   = alpha  [:, :, start:end]
        bt   = beta   [:, :, start:end]

        chunk_out = torch.zeros(B, H, T, d, device=k_tilde.device, dtype=k_tilde.dtype)

        for t in range(T):
            retrieved = torch.einsum('bhij,bhj->bhi', mem, kt[:, :, t, :])
            delta     = vt[:, :, t, :] - retrieved
            outer     = torch.einsum('bhi,bhj->bhij', delta, kt[:, :, t, :])
            a         = at[:, :, t].unsqueeze(-1).unsqueeze(-1)
            b         = bt[:, :, t].unsqueeze(-1).unsqueeze(-1)
            mem       = a * mem + b * outer
            chunk_out[:, :, t, :] = torch.einsum('bhij,bhj->bhi', mem, qt[:, :, t, :])

        out_chunks.append(chunk_out)
        mem = mem.detach()

    return torch.cat(out_chunks, dim=2), mem


def _fla_gated_chunk_scan(
    k_tilde   : torch.Tensor,
    v         : torch.Tensor,
    q_tilde   : torch.Tensor,
    beta      : torch.Tensor,
    alpha     : torch.Tensor,
    mem_init  : torch.Tensor,
    chunk_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_dtype    = q_tilde.dtype
    scale         = math.sqrt(q_tilde.shape[-1])
    alpha_clamped = alpha.clamp(min=1e-4, max=1.0 - 1e-4)
    g             = torch.log(alpha_clamped)

    q  = (q_tilde * scale).to(torch.bfloat16).permute(0, 2, 1, 3).contiguous()
    k  = k_tilde.to(torch.bfloat16).permute(0, 2, 1, 3).contiguous()
    v_ = v.to(torch.bfloat16).permute(0, 2, 1, 3).contiguous()
    g_ = g.to(torch.bfloat16).permute(0, 2, 1).contiguous()
    b  = beta.to(torch.bfloat16).permute(0, 2, 1).contiguous()
    h0 = mem_init.to(torch.bfloat16).contiguous()

    out, mem_final = _fla_gated_rule(
        q                       = q,
        k                       = k,
        v                       = v_,
        g                       = g_,
        beta                    = b,
        initial_state           = h0,
        output_final_state      = True,
        use_qk_l2norm_in_kernel = False,
        chunk_size              = chunk_size,
    )
    out = out.permute(0, 2, 1, 3)
    return out.to(orig_dtype), mem_final.to(orig_dtype)


def _fla_delta_approx_chunk_scan(
    k_tilde   : torch.Tensor,
    v         : torch.Tensor,
    q_tilde   : torch.Tensor,
    beta      : torch.Tensor,
    alpha     : torch.Tensor,
    mem_init  : torch.Tensor,
    chunk_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = q_tilde.dtype
    q          = q_tilde.to(torch.bfloat16).permute(0, 2, 1, 3).contiguous()
    k          = k_tilde.to(torch.bfloat16).permute(0, 2, 1, 3).contiguous()
    v_         = (v * alpha.unsqueeze(-1)).to(torch.bfloat16).permute(0, 2, 1, 3).contiguous()
    b          = beta.to(torch.bfloat16).permute(0, 2, 1).contiguous()
    h0         = mem_init.to(torch.bfloat16)

    out, mem_final = _fla_delta_rule(
        q                  = q,
        k                  = k,
        v                  = v_,
        beta               = b,
        initial_state      = h0,
        output_final_state = True,
    )
    out = out.permute(0, 2, 1, 3)
    return out.to(orig_dtype), mem_final.to(orig_dtype)


def _delta_rule_scan(
    k_tilde   : torch.Tensor,
    v         : torch.Tensor,
    q_tilde   : torch.Tensor,
    beta      : torch.Tensor,
    alpha     : torch.Tensor,
    mem_init  : torch.Tensor,
    chunk_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not FORCE_PYTHON_SCAN and _HAS_FLA_GATED and k_tilde.is_cuda:
        try:
            return _fla_gated_chunk_scan(k_tilde, v, q_tilde, beta, alpha, mem_init, chunk_size)
        except Exception as e:
            print(f'  ⚠️  FLA gated kernel error : {e} — fallback Python')
    return _python_chunk_scan(k_tilde, v, q_tilde, beta, alpha, mem_init, chunk_size)


class GDNNaylisAttention(nn.Module):

    def __init__(
        self,
        embed_dim   : int,
        num_heads   : int,
        n_kv_heads  : Optional[int] = None,
        rel_rank    : int   = 48,
        dropout     : float = 0.0,
        use_qk_norm : bool  = True,
        conv_kernel : int   = 4,
        use_rope    : bool  = False,
        chunk_size  : int   = 256,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.rel_rank   = rel_rank
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else num_heads
        self.chunk_size = chunk_size

        assert num_heads % self.n_kv_heads == 0
        self.kv_dim = self.n_kv_heads * self.head_dim

        self.q_proj   = nn.Linear(embed_dim, embed_dim,   bias=False)
        self.k_proj   = nn.Linear(embed_dim, self.kv_dim, bias=False)
        self.v_proj   = nn.Linear(embed_dim, self.kv_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim,   bias=False)

        self.q_conv = ShortConv1d(embed_dim,   conv_kernel)
        self.k_conv = ShortConv1d(self.kv_dim, conv_kernel)
        self.v_conv = ShortConv1d(self.kv_dim, conv_kernel)

        self.forget_gate = nn.Linear(embed_dim, num_heads, bias=True)
        self.learn_gate  = nn.Linear(embed_dim, num_heads, bias=True)
        nn.init.constant_(self.forget_gate.bias, 2.0)
        nn.init.constant_(self.learn_gate.bias,  0.0)

        self.rel_q_proj  = nn.Linear(embed_dim, num_heads * rel_rank, bias=False)
        self.rel_k_proj  = nn.Linear(embed_dim, num_heads * rel_rank, bias=False)
        self.graph_scale = nn.Parameter(torch.zeros(num_heads))

        nn.init.normal_(self.rel_q_proj.weight, std=0.01)
        nn.init.normal_(self.rel_k_proj.weight, std=0.01)

        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = self.k_norm = None

        self.o_norm  = RMSNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _expand_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_kv_heads == self.num_heads:
            return x
        return x.repeat_interleave(self.num_heads // self.n_kv_heads, dim=1)

    def forward(
        self,
        x        : torch.Tensor,
        state    : Optional[GDNState] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[GDNState]]:

        B, S, D = x.shape
        H, d, R = self.num_heads, self.head_dim, self.rel_rank

        cache_q = cache_k = cache_v = None
        if state is not None and state.conv_caches is not None:
            cache_q, cache_k, cache_v = state.conv_caches

        q_raw, cache_q_new = self.q_conv(self.q_proj(x), cache_q)
        k_raw, cache_k_new = self.k_conv(self.k_proj(x), cache_k)
        v_raw, cache_v_new = self.v_conv(self.v_proj(x), cache_v)

        q = q_raw.view(B, S, H,               d).permute(0, 2, 1, 3)
        k = k_raw.view(B, S, self.n_kv_heads, d).permute(0, 2, 1, 3)
        v = v_raw.view(B, S, self.n_kv_heads, d).permute(0, 2, 1, 3)

        k = self._expand_kv(k)
        v = self._expand_kv(v)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        R_q = self.rel_q_proj(x).view(B, S, H, R).permute(0, 2, 1, 3)
        R_k = self.rel_k_proj(x).view(B, S, H, R).permute(0, 2, 1, 3)

        if R < d:
            pad = torch.zeros(B, H, S, d - R, device=x.device, dtype=x.dtype)
            R_q = torch.cat([R_q, pad], dim=-1)
            R_k = torch.cat([R_k, pad], dim=-1)
        elif R > d:
            R_q = R_q[..., :d]
            R_k = R_k[..., :d]

        lam     = self.graph_scale.view(1, H, 1, 1).abs()
        k_tilde = F.normalize(k + lam * R_k, dim=-1)
        q_tilde = q + lam * R_q

        alpha = torch.sigmoid(self.forget_gate(x)).permute(0, 2, 1)
        beta  = torch.sigmoid(self.learn_gate(x)).permute(0, 2, 1)

        mem_init = (
            state.S.to(x.dtype) if state is not None
            else torch.zeros(B, H, d, d, device=x.device, dtype=x.dtype)
        )

        out, mem_final = _delta_rule_scan(
            k_tilde    = k_tilde,
            v          = v,
            q_tilde    = q_tilde,
            beta       = beta,
            alpha      = alpha,
            mem_init   = mem_init,
            chunk_size = self.chunk_size,
        )

        out = out.permute(0, 2, 1, 3).contiguous().view(B, S, D)
        out = self.o_norm(out)
        out = self.out_proj(out)
        out = self.dropout(out)

        new_state = GDNState(mem_final, [cache_q_new, cache_k_new, cache_v_new]) if use_cache else None

        return out, new_state