#!/usr/bin/env python3

import os
os.environ["TORCHINDUCTOR_CACHE_DIR"]      = "./CompileCache"
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
os.makedirs("./CompileCache", exist_ok=True)

import sys
import time
import math
import json
import gc
import traceback
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.set_float32_matmul_precision('high')

_root = os.path.dirname(__file__)
sys.path.append(os.path.join(_root, 'Core', 'Model'))
sys.path.append(os.path.join(_root, 'Core', 'Attention'))
sys.path.append(os.path.join(_root, 'Core', 'FeedForward'))
sys.path.append(os.path.join(_root, 'Core', 'TransformerBlock'))

from GNT import GNT
from attention import NaylisAttention


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--no-compile',   action='store_true')
    p.add_argument('--compile-mode', default='default',
                   choices=['default', 'reduce-overhead', 'max-autotune'])
    p.add_argument('--batch-size',   type=int, default=2)
    p.add_argument('--grad-accum',   type=int, default=8)
    p.add_argument('--seq-len',      type=int, default=1024)
    return p.parse_args()


ARGS   = get_args()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CONFIG = {
    'vocab_size'            : None,
    'embed_dim'             : 768,
    'num_heads'             : 12,
    'n_kv_heads'            : 4,
    'num_layers'            : 18,
    'rel_rank'              : 32,
    'max_seq_len'           : ARGS.seq_len,
    'dropout'               : 0.0,
    'attn_every_n_layers'   : 4,
    'use_qk_norm'           : True,
    'conv_kernel'           : 4,
    'batch_size'            : ARGS.batch_size,
    'gradient_accumulation' : ARGS.grad_accum,
    'max_grad_norm'         : 1.0,
    'learning_rate'         : 4e-4,
    'weight_decay'          : 0.1,
    'adam_beta1'            : 0.9,
    'adam_beta2'            : 0.95,
    'adam_eps'              : 1e-8,
    'warmup_ratio'          : 0.03,
    'decay_ratio'           : 0.15,
    'min_lr_ratio'          : 0.1,
    'data_dir'              : './data',
    'tokenizer_dir'         : './data/tokenizer_gnt',
    'val_split_tokens'      : 10_000_000,
    'val_every_steps'       : 500,
    'val_batches'           : 50,
    'save_every_steps'      : 2_000,
    'naylis_log_every'      : 1_000,
    'ckpt_path'             : './Model/gnt_pretrain.pt',
    'use_compile'           : not ARGS.no_compile,
    'compile_mode'          : ARGS.compile_mode,
    'num_workers'           : 1,
    'loss_mavg_window'      : 20,
    'loss_ema_alpha'        : 0.98,
    # ── Dry run ───────────────────────────────────────────────────────────────
    # True  → entraîne sur ./data/dry_chunk uniquement (1 pass, pas de save ckpt)
    # False → entraîne sur chunk_000 … chunk_XXX (curriculum complet)
    'dry_run'               : False,
}

print('=' * 70)
print('  GNT v1 — Pretrain')
print('=' * 70)
if DEVICE == 'cuda':
    print(f'  GPU  : {torch.cuda.get_device_name(0)}')
    print(f'  VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB')
    cap = torch.cuda.get_device_capability()
    print(f'  SM   : {cap[0]}{cap[1]}')
print(f'  embed={CONFIG["embed_dim"]}  layers={CONFIG["num_layers"]}  '
      f'heads={CONFIG["num_heads"]}  kv={CONFIG["n_kv_heads"]}  '
      f'rel_rank={CONFIG["rel_rank"]}  seq={CONFIG["max_seq_len"]}')
print(f'  attn_every={CONFIG["attn_every_n_layers"]} layers')
print(f'  dry_run={CONFIG["dry_run"]}')


print('\nTokenizer...')
tok_dir = Path(CONFIG['tokenizer_dir'])
if tok_dir.exists():
    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
    print(f'  Tokenizer GNT chargé : vocab={len(tokenizer)}')
else:
    print('  ⚠️  Tokenizer GNT non trouvé — utilisation cosmo2 de base')
    tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/cosmo2-tokenizer')

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

CONFIG['vocab_size'] = len(tokenizer)
EOS_ID         = tokenizer.eos_token_id
THINK_START_ID = tokenizer.convert_tokens_to_ids('<think>')
THINK_END_ID   = tokenizer.convert_tokens_to_ids('</think>')
CODE_START_ID  = tokenizer.convert_tokens_to_ids('<code>')
CODE_END_ID    = tokenizer.convert_tokens_to_ids('</code>')
OUT_START_ID   = tokenizer.convert_tokens_to_ids('<output>')
OUT_END_ID     = tokenizer.convert_tokens_to_ids('</output>')
print(f'  vocab={len(tokenizer)}  eos={EOS_ID}')
print(f'  <think>={THINK_START_ID}  </think>={THINK_END_ID}')
print(f'  <code>={CODE_START_ID}  </code>={CODE_END_ID}')
print(f'  <output>={OUT_START_ID}  </output>={OUT_END_ID}')


def scan_chunks(data_dir: str, dry_run: bool) -> list:
    """
    dry_run=True  → retourne uniquement ./data/dry_chunk/tokens.npy
    dry_run=False → retourne chunk_000, chunk_001, … dans l'ordre
    """
    base = Path(data_dir)
    if dry_run:
        f = base / 'dry_chunk' / 'tokens.npy'
        if not f.exists():
            return []
        arr = np.load(str(f), mmap_mode='r')
        return [{'path': str(f), 'tokens': len(arr)}]

    chunks = []
    for d in sorted(base.iterdir()):
        if not d.is_dir() or d.name == 'dry_chunk':
            continue
        f = d / 'tokens.npy'
        if f.exists():
            arr = np.load(str(f), mmap_mode='r')
            chunks.append({'path': str(f), 'tokens': len(arr)})
    return chunks


class ChunkDataset:
    def __init__(self, path: str, val_tokens: int, seq_len: int, batch_size: int):
        self.arr        = np.load(path, mmap_mode='r')
        n               = len(self.arr)
        self.train_arr  = self.arr[:n - val_tokens]
        self.val_arr    = self.arr[n - val_tokens:]
        self.seq_len    = seq_len
        self.batch_size = batch_size

        # Pas de shuffle ici — déjà fait au download (shuffle_chunk, seq_len=1024).
        n_train     = len(self.train_arr)
        n_seqs      = n_train // (seq_len + 1)
        self.starts = np.arange(n_seqs) * (seq_len + 1)
        self.pos    = 0
        self.epoch  = 0
        print(f'  séquences={n_seqs:,}')

    @property
    def n_train_batches(self):
        return len(self.starts) // self.batch_size

    def next_batch(self, device) -> tuple[torch.Tensor, torch.Tensor]:
        L   = self.seq_len
        B   = self.batch_size
        ids = []
        for _ in range(B):
            if self.pos >= len(self.starts):
                self.epoch += 1
                self.pos = 0
            s = self.starts[self.pos]
            ids.append(self.train_arr[s : s + L + 1].astype(np.int64))
            self.pos += 1
        ids = np.stack(ids)
        x   = torch.from_numpy(ids[:, :-1]).to(device)
        y   = torch.from_numpy(ids[:, 1:]).to(device)
        return x, y

    def val_batches_iter(self, device, max_batches: int):
        L   = self.seq_len
        B   = self.batch_size
        arr = self.val_arr
        pos = 0
        n   = 0
        while pos + L + 1 <= len(arr) and n < max_batches:
            batch = []
            for _ in range(B):
                if pos + L + 1 > len(arr): break
                batch.append(arr[pos:pos+L+1].astype(np.int64))
                pos += L + 1
            if len(batch) < B: break
            ids = np.stack(batch)
            x   = torch.from_numpy(ids[:, :-1]).to(device)
            y   = torch.from_numpy(ids[:, 1:]).to(device)
            yield x, y
            n += 1


class WSDScheduler:
    def __init__(self, optimizers, max_lr, total_steps,
                 warmup_ratio=0.03, decay_ratio=0.15, min_lr_ratio=0.1):
        self.opts   = optimizers if isinstance(optimizers, list) else [optimizers]
        self.max_lr = max_lr
        self.min_lr = max_lr * min_lr_ratio
        self.total  = total_steps
        self.warmup = max(1, int(total_steps * warmup_ratio))
        self.decay  = max(1, int(total_steps * decay_ratio))
        self.stable = max(0, total_steps - self.warmup - self.decay)
        self.step_n = 0

    def get_lr(self):
        s = self.step_n
        if s < self.warmup:
            return self.max_lr * (s / self.warmup)
        elif s < self.warmup + self.stable:
            return self.max_lr
        else:
            d = s - self.warmup - self.stable
            p = min(d / max(self.decay, 1), 1.0)
            return self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * p))

    def step(self):
        lr = self.get_lr()
        self.step_n += 1
        for opt in self.opts:
            for pg in opt.param_groups:
                if pg.get('is_muon'):
                    pg['lr'] = lr * 50.0
                elif pg.get('is_adamw_slow'):
                    pg['lr'] = lr * 0.2
                else:
                    pg['lr'] = lr
        return lr

    def state_dict(self):         return {'step_n': self.step_n}
    def load_state_dict(self, d): self.step_n = d.get('step_n', 0)


def _zeropower_via_newtonschulz5(G, steps=5):
    assert G.ndim >= 2
    a, b, c    = 3.4445, -4.7750, 2.0315
    orig_shape = G.shape
    if G.ndim > 2:
        G = G.reshape(G.shape[0], -1)
    X = G.bfloat16() / (G.norm() + 1e-7)
    if G.size(0) > G.size(1): X = X.mT
    for _ in range(steps):
        A = X @ X.mT; B = b*A + c*(A@A); X = a*X + B@X
    if G.size(0) > G.size(1): X = X.mT
    return X.to(G.dtype).reshape(orig_shape)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.0, use_mars=True, mars_gamma=0.025,
                 mom_warmup_steps=300):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                     ns_steps=ns_steps, weight_decay=weight_decay,
                                     use_mars=use_mars, mars_gamma=mars_gamma,
                                     mom_warmup_steps=mom_warmup_steps))
        self._global_step = 0

    def state_dict(self):
        d = super().state_dict()
        d['_global_step'] = self._global_step
        return d

    def load_state_dict(self, state_dict):
        state_dict        = dict(state_dict)
        self._global_step = state_dict.pop('_global_step', 0)
        super().load_state_dict(state_dict)

    @torch.no_grad()
    def step(self):
        self._global_step += 1
        for group in self.param_groups:
            lr, nest     = group['lr'], group['nesterov']
            ns, wd       = group['ns_steps'], group['weight_decay']
            use_mars, mg = group.get('use_mars', True), group.get('mars_gamma', 0.025)
            warmup_steps = group.get('mom_warmup_steps', 300)
            target_mom   = group['momentum']
            mom = (
                0.85 + (target_mom - 0.85) * (self._global_step / warmup_steps)
                if self._global_step < warmup_steps else target_mom
            )

            for p in group['params']:
                if p.grad is None or p.grad.ndim < 2: continue
                g = p.grad
                s = self.state[p]
                if use_mars:
                    if 'prev_grad' not in s:
                        s['prev_grad'] = torch.zeros_like(g)
                    prev = s['prev_grad']
                    c_t  = torch.clamp(
                        (mg / (1-mg)) * (g.norm()+1e-8) / (prev.norm()+1e-8), max=1.0)
                    g    = g + c_t * (g - prev)
                    s['prev_grad'].copy_(p.grad)
                if 'buf' not in s: s['buf'] = torch.zeros_like(g)
                buf = s['buf']
                buf.mul_(mom).add_(g)
                g   = (g + mom*buf) if nest else buf
                g   = _zeropower_via_newtonschulz5(g, steps=ns)
                d_out, d_in = g.size(0), g.size(1)
                g = g * (d_in ** 0.5) * max(d_out / d_in, 1.0) ** 0.5
                if wd: p.mul_(1. - lr*wd)
                p.add_(g, alpha=-lr)


def configure_optimizers(model, cfg):
    EXCLUDE       = {'token_embeddings.weight', 'output_head.weight'}
    muon_p, adamw_2d, adamw_emb, adamw_slow = [], [], [], []
    _muon_exclude = ('norm', 'embed', 'rel_q_proj', 'rel_k_proj', 'graph_scale')
    _slow_kw      = ('graph_scale', 'norm', 'bias')

    for pn, p in model.named_parameters():
        if not p.requires_grad: continue
        if pn in EXCLUDE:
            adamw_emb.append(p); continue
        if (p.ndim >= 2
                and pn.startswith('blocks.')
                and not any(x in pn.lower() for x in _muon_exclude)):
            muon_p.append(p)
        elif p.ndim >= 2:
            adamw_2d.append(p)
        elif any(x in pn.lower() for x in _slow_kw):
            adamw_slow.append(p)
        else:
            adamw_slow.append(p)

    lr_base = cfg['learning_rate']
    lr_slow = lr_base * 0.2

    muon   = Muon([{'params': muon_p, 'is_muon': True}],
                  lr=lr_base, momentum=0.95, ns_steps=5,
                  mom_warmup_steps=300, weight_decay=cfg['weight_decay'])
    _fused = torch.cuda.is_available()
    adamw  = torch.optim.AdamW(
        [
            {'params': adamw_2d,   'lr': lr_base, 'weight_decay': cfg['weight_decay'],
             'is_adamw_base': True},
            {'params': adamw_emb,  'lr': lr_base, 'weight_decay': 0.0,
             'is_adamw_base': True},
            {'params': adamw_slow, 'lr': lr_slow, 'weight_decay': 0.0,
             'is_adamw_slow': True},
        ],
        betas=(cfg['adam_beta1'], cfg['adam_beta2']),
        eps=cfg['adam_eps'],
        fused=_fused,
    )
    n_muon = sum(p.numel() for p in muon_p)
    n_2d   = sum(p.numel() for p in adamw_2d)
    n_emb  = sum(p.numel() for p in adamw_emb)
    n_slow = sum(p.numel() for p in adamw_slow)
    print(f'  Muon+MARS  : {n_muon/1e6:.2f}M  lr_effectif={lr_base*50:.2e}')
    print(f'  AdamW 2D   : {n_2d/1e6:.2f}M  lr={lr_base:.2e}')
    print(f'  AdamW emb  : {n_emb/1e6:.2f}M  lr={lr_base:.2e}')
    print(f'  AdamW slow : {n_slow/1e6:.2f}M  lr={lr_slow:.2e}')
    return muon, adamw


def save_ckpt(model, optimizers, scheduler, meta, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    m           = model._orig_mod if hasattr(model, '_orig_mod') else model
    muon, adamw = optimizers
    torch.save({
        'model_state_dict'    : m.state_dict(),
        'muon_state_dict'     : muon.state_dict(),
        'adamw_state_dict'    : adamw.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'meta'                : meta,
        'config'              : CONFIG,
    }, path + '.tmp')
    os.replace(path + '.tmp', path)
    print(f'  💾 SAVE  step={meta["global_step"]:,}  [{path}]')


def load_ckpt(path):
    if not os.path.exists(path): return None
    return torch.load(path, map_location='cpu', weights_only=False)


@torch.no_grad()
def validate(model, ds, max_batches):
    model.eval()
    total, n = 0.0, 0
    ae  = DEVICE == 'cuda'
    adt = torch.bfloat16 if ae else torch.float32
    for x, y in ds.val_batches_iter(DEVICE, max_batches):
        with torch.amp.autocast(DEVICE, dtype=adt, enabled=ae):
            _, loss, _ = model(x, targets=y)
        total += loss.item(); n += 1
    model.train()
    val_loss = total / max(n, 1)
    return val_loss, math.exp(min(val_loss, 20.0))


def log_graph_scale(model, step, pbar=None):
    raw         = model._orig_mod if hasattr(model, '_orig_mod') else model
    layer_stats = []
    for b in raw.blocks:
        attn = getattr(b, 'attention', None)
        if attn is None or not hasattr(attn, 'graph_scale'):
            continue
        gs = attn.graph_scale.detach().abs().float()
        layer_stats.append({
            'mean': gs.mean().item(),
            'max' : gs.max().item(),
            'min' : gs.min().item(),
            'std' : gs.std().item() if gs.numel() > 1 else 0.0,
        })

    if not layer_stats:
        return layer_stats

    means      = [s['mean'] for s in layer_stats]
    global_avg = sum(means) / len(means)
    global_max = max(s['max'] for s in layer_stats)
    global_min = min(s['min'] for s in layer_stats)
    layer_str  = ', '.join(f'{s["mean"]:.4f}\xb1{s["std"]:.4f}' for s in layer_stats)
    msg        = (f'  [graph_scale step={step:,}] '
                  f'avg={global_avg:.5f}  max={global_max:.5f}  min={global_min:.5f}  '
                  f'layers=[{layer_str}]')

    warn_msgs = []
    if step >= 1000:
        for i, s in enumerate(layer_stats):
            if s['max'] > 50.0:
                warn_msgs.append(f'    ⚠️  graph_scale EXPLOSION  layer={i}  max={s["max"]:.3f}')
            if s['mean'] < 1e-10:
                warn_msgs.append(f'    ⚠️  graph_scale EFFONDREMENT  layer={i}  mean={s["mean"]:.5f}')
            if s['std'] > 10.0:
                warn_msgs.append(f'    ⚠️  graph_scale VARIANCE ÉLEVÉE  layer={i}  std={s["std"]:.3f}')

    def _write(m):
        if pbar is not None: pbar.write(m)
        else: print(m)

    _write(msg)
    for w in warn_msgs:
        _write(w)

    return layer_stats


def _moving_average(values: list, window: int) -> list:
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start:i+1]) / (i - start + 1))
    return result


def _ema(values: list, alpha: float) -> list:
    result, ema = [], None
    for v in values:
        ema = v if ema is None else alpha * ema + (1 - alpha) * v
        result.append(ema)
    return result


def _style_ax(ax):
    ax.set_facecolor('#1a1d27')
    ax.tick_params(colors='#cccccc')
    ax.xaxis.label.set_color('#cccccc')
    ax.yaxis.label.set_color('#cccccc')
    ax.title.set_color('#ffffff')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333344')
    ax.grid(True, color='#2a2d3a', linewidth=0.5)


def plot_loss(history: list, path: str = './Model/gnt_loss.png', pbar=None):
    if not history:
        return
    window    = CONFIG.get('loss_mavg_window', 20)
    alpha     = CONFIG.get('loss_ema_alpha', 0.98)
    steps     = [h['step']     for h in history]
    val_loss  = [h['val_loss'] for h in history]
    has_train = all('train_loss' in h for h in history)

    path_window = path.replace('.png', '_window.png')
    path_ema    = path.replace('.png', '_ema.png')

    for mode in ('window', 'ema'):
        fig, ax = plt.subplots(figsize=(13, 5))
        fig.patch.set_facecolor('#0e1117')
        _style_ax(ax)

        if has_train:
            train_loss = [h['train_loss'] for h in history]
            ax.plot(steps, train_loss, color='#ffb74d', linewidth=0.7,
                    alpha=0.2, label='_nolegend_')
            if mode == 'window':
                smoothed = _moving_average(train_loss, window)
                label    = f'Train Loss (mavg {window})'
            else:
                smoothed = _ema(train_loss, alpha)
                label    = f'Train Loss (EMA α={alpha})'
            ax.plot(steps, smoothed, color='#ffb74d', linewidth=1.8, label=label)

        ax.plot(steps, val_loss, color='#4fc3f7', linewidth=0.7,
                alpha=0.2, label='_nolegend_')

        has_val_smooth = all('avg_val_loss' in h and 'val_loss_ema' in h for h in history)
        if has_val_smooth:
            if mode == 'window':
                val_smoothed = [h['avg_val_loss'] for h in history]
                val_label    = f'Val Loss (mavg {window})'
            else:
                val_smoothed = [h['val_loss_ema'] for h in history]
                val_label    = f'Val Loss (EMA α={alpha})'
            ax.plot(steps, val_smoothed, color='#4fc3f7', linewidth=1.8, label=val_label)
        else:
            ax.plot(steps, val_loss, color='#4fc3f7', linewidth=1.8, label='Val Loss')

        all_y = list(val_loss)
        if has_train:
            all_y += [h['train_loss'] for h in history]
        if has_val_smooth:
            all_y += val_smoothed
        y_min, y_max = min(all_y), max(all_y)
        margin = max((y_max - y_min) * 0.1, 0.05)
        ax.set_ylim(y_min - margin, y_max + margin)

        ax.set_ylabel('Cross-Entropy Loss')
        ax.set_xlabel('Step')
        suffix = 'Window' if mode == 'window' else 'EMA'
        ax.set_title(f'GNT Pretrain — Train & Val Loss ({suffix})')
        ax.legend(facecolor='#1a1d27', labelcolor='#cccccc')

        plt.tight_layout()
        out = path_window if mode == 'window' else path_ema
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

    msg = f'  📊 Loss graphs → {path_window}  |  {path_ema}'
    if pbar is not None: pbar.write(msg)
    else: print(msg)


def main():
    dry_run = CONFIG['dry_run']

    print('\n' + '=' * 70)
    print(f'  SCAN CHUNKS  {"[DRY RUN]" if dry_run else ""}')
    print('=' * 70)

    chunks = scan_chunks(CONFIG['data_dir'], dry_run)
    if not chunks:
        if dry_run:
            print('  ❌ dry_chunk introuvable dans', CONFIG['data_dir'])
            print('  → Lance : python3 dataset_gnt.py')
        else:
            print('  ❌ Aucun chunk trouvé dans', CONFIG['data_dir'])
            print('  → Lance : python3 dataset_gnt.py --phase 1')
        return

    print(f'  {len(chunks)} chunk(s) :')
    for i, c in enumerate(chunks):
        print(f'    [{i}] {Path(c["path"]).parent.name} : {c["tokens"]/1e9:.3f}B tokens')

    total_train_tokens = sum(c['tokens'] - CONFIG['val_split_tokens'] for c in chunks)
    steps_per_token    = 1 / (CONFIG['batch_size'] * CONFIG['max_seq_len'])
    total_steps        = int(total_train_tokens * steps_per_token / CONFIG['gradient_accumulation'])
    print(f'  Total steps estimés : {total_steps:,}')

    print('\n' + '=' * 70)
    print('  CRÉATION MODÈLE GNT')
    print('=' * 70)

    dtype = torch.bfloat16 if DEVICE == 'cuda' else torch.float32
    model = GNT(
        vocab_size          = CONFIG['vocab_size'],
        embed_dim           = CONFIG['embed_dim'],
        num_heads           = CONFIG['num_heads'],
        n_kv_heads          = CONFIG['n_kv_heads'],
        num_layers          = CONFIG['num_layers'],
        rel_rank            = CONFIG['rel_rank'],
        max_seq_len         = CONFIG['max_seq_len'],
        dropout             = CONFIG['dropout'],
        attn_every_n_layers = CONFIG['attn_every_n_layers'],
        use_qk_norm         = CONFIG['use_qk_norm'],
        conv_kernel         = CONFIG['conv_kernel'],
    ).to(dtype).to(DEVICE)

    p = model.count_parameters()
    print(f'  Params total : {p["total_M"]}M')
    print(f'  GDN          : {p["gdn_M"]}M')
    print(f'  Naylis       : {p["naylis_K"]}K = {p["naylis_pct"]}')
    print(f'  dtype        : {dtype}')

    if DEVICE == 'cuda':
        non_bf16 = [(n, prm.dtype) for n, prm in model.named_parameters()
                    if prm.dtype not in (torch.bfloat16, torch.int8, torch.int32, torch.long)]
        if non_bf16:
            print(f'  ⚠️  {len(non_bf16)} param(s) NON bf16 après init :')
            for n, dt in non_bf16[:10]:
                print(f'       {n}: {dt}')
        else:
            print('  ✅ Tous les poids sont en bf16')

    if CONFIG['use_compile'] and DEVICE == 'cuda':
        print('\ntorch.compile...')
        import torch._dynamo as _dynamo
        _dynamo.config.cache_size_limit = 256
        _dynamo.config.suppress_errors  = True
        model = torch.compile(model, mode=CONFIG['compile_mode'])
        print('  OK')

    raw  = model._orig_mod if hasattr(model, '_orig_mod') else model
    opts = configure_optimizers(raw, CONFIG)
    muon, adamw = opts

    global_step = 0
    start_chunk = 0

    # En dry_run on ne charge pas de checkpoint et on ne sauvegarde pas
    cp = None if dry_run else load_ckpt(CONFIG['ckpt_path'])

    scheduler = WSDScheduler(
        list(opts), CONFIG['learning_rate'], total_steps,
        CONFIG['warmup_ratio'], CONFIG['decay_ratio'], CONFIG['min_lr_ratio'],
    )
    if cp is not None:
        raw.load_state_dict(cp['model_state_dict'], strict=False)
        muon.load_state_dict(cp.get('muon_state_dict', {}))
        adamw.load_state_dict(cp.get('adamw_state_dict', {}))
        global_step = cp.get('meta', {}).get('global_step', 0)
        start_chunk = cp.get('meta', {}).get('chunk_idx', 0)
        scheduler.load_state_dict(cp.get('scheduler_state_dict', {}))
        print(f'\n  Reprise depuis step {global_step:,}  chunk_idx={start_chunk}')
        print(f'  LR reprise : {scheduler.get_lr():.2e}')

    print('\n' + '=' * 70)
    print(f'  TRAINING START — {total_steps:,} steps'
          + ('  [DRY RUN — 1 pass, pas de save]' if dry_run else ''))
    print('=' * 70)

    ae  = DEVICE == 'cuda'
    adt = torch.bfloat16 if ae else torch.float32

    model.train()
    history_path    = './Model/gnt_history.json'
    gs_history_path = './Model/gnt_graph_scale_history.json'
    history         = []
    gs_history      = []
    if cp is not None and os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        print(f'  History rechargé : {len(history)} points de val')
    if cp is not None and os.path.exists(gs_history_path):
        with open(gs_history_path) as f:
            gs_history = json.load(f)
        print(f'  graph_scale history rechargé : {len(gs_history)} points')

    loss_window  = CONFIG['loss_mavg_window']
    loss_alpha   = CONFIG['loss_ema_alpha']

    if history:
        past_train   = [h['train_loss'] for h in history if 'train_loss' in h]
        past_val     = [h['val_loss']   for h in history]
        loss_buf     = past_train[-loss_window:]
        val_loss_buf = past_val[-loss_window:]
        loss_ema     = _ema(past_train, loss_alpha)[-1] if past_train else None
        val_loss_ema = _ema(past_val,   loss_alpha)[-1]
    else:
        loss_buf     = []
        loss_ema     = None
        val_loss_buf = []
        val_loss_ema = None

    t0 = time.time()

    for chunk_idx, chunk_info in enumerate(chunks):
        if not dry_run and chunk_idx < start_chunk:
            print(f'  ⏩ Chunk {chunk_idx} déjà traité — skip')
            continue

        print(f'\n{"="*70}')
        print(f'  Chunk : {chunk_info["path"]}')
        print('=' * 70)

        ds = ChunkDataset(
            chunk_info['path'],
            CONFIG['val_split_tokens'],
            CONFIG['max_seq_len'],
            CONFIG['batch_size'],
        )
        if cp is not None and chunk_idx == start_chunk:
            ds.pos   = cp.get('meta', {}).get('ds_pos', 0)
            ds.epoch = cp.get('meta', {}).get('ds_epoch', 0)
            if ds.pos > 0:
                print(f'  Reprise position dataset : pos={ds.pos:,}  epoch={ds.epoch}')
        print(f'  batches={ds.n_train_batches:,}')

        acc_steps   = 0
        acc_loss    = 0.0
        n_remaining = ds.n_train_batches - (ds.pos // ds.batch_size)

        muon.zero_grad(set_to_none=True)
        adamw.zero_grad(set_to_none=True)

        pbar = tqdm(
            total         = ds.n_train_batches,
            desc          = '  GNT',
            dynamic_ncols = True,
            colour        = 'cyan',
            initial       = ds.pos,
        )

        for _ in range(n_remaining):
            # En dry_run on fait exactement 1 pass (epoch 0 seulement)
            if dry_run and ds.epoch > 0:
                break

            x, y = ds.next_batch(DEVICE)

            with torch.amp.autocast(DEVICE, dtype=adt, enabled=ae):
                _, loss, _ = model(x, targets=y)
                loss       = loss / CONFIG['gradient_accumulation']

            loss.backward()
            acc_loss  += loss.item()
            acc_steps += 1

            if acc_steps >= CONFIG['gradient_accumulation']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
                muon.step()
                adamw.step()
                lr = scheduler.step()
                muon.zero_grad(set_to_none=True)
                adamw.zero_grad(set_to_none=True)

                global_step += 1
                tl           = acc_loss
                acc_loss     = 0.0
                acc_steps    = 0

                loss_buf.append(tl)
                if len(loss_buf) > loss_window:
                    loss_buf.pop(0)
                avg_loss  = sum(loss_buf) / len(loss_buf)
                loss_ema  = tl if loss_ema is None else loss_alpha * loss_ema + (1 - loss_alpha) * tl
                train_ppl = math.exp(min(avg_loss, 20.0))

                pbar.set_postfix(
                    avg_loss     = f'{avg_loss:.4f}',
                    avg_loss_ema = f'{loss_ema:.4f}',
                    ppl          = f'{train_ppl:.1f}',
                    lr           = f'{lr:.2e}',
                    step         = global_step,
                )

                if global_step % CONFIG['val_every_steps'] == 0:
                    vl, vp  = validate(model, ds, CONFIG['val_batches'])
                    elapsed = (time.time() - t0) / 60

                    val_loss_buf.append(vl)
                    if len(val_loss_buf) > loss_window:
                        val_loss_buf.pop(0)
                    avg_val_loss = sum(val_loss_buf) / len(val_loss_buf)
                    val_loss_ema = (vl if val_loss_ema is None
                                    else loss_alpha * val_loss_ema + (1 - loss_alpha) * vl)

                    pbar.write(
                        f'  [val  step={global_step:,}] '
                        f'loss={vl:.4f}  avg={avg_val_loss:.4f}  ema={val_loss_ema:.4f}'
                        f'  ppl={vp:.2f}  {elapsed:.1f}min'
                    )
                    history.append({
                        'step'        : global_step,
                        'val_loss'    : vl,
                        'val_ppl'     : vp,
                        'avg_val_loss': avg_val_loss,
                        'val_loss_ema': val_loss_ema,
                        'train_loss'  : avg_loss,
                        'train_ppl'   : train_ppl,
                    })
                    plot_loss(history, pbar=pbar)
                    train_mavg = _moving_average([h['train_loss'] for h in history], loss_window)[-1]
                    train_ema_ = _ema([h['train_loss'] for h in history], loss_alpha)[-1]
                    pbar.write(
                        f'  📈 Stats  '
                        f'train_mavg={train_mavg:.4f}  train_ema={train_ema_:.4f}  '
                        f'val_mavg={avg_val_loss:.4f}  val_ema={val_loss_ema:.4f}'
                    )

                if global_step == 100 or global_step % CONFIG['naylis_log_every'] == 0:
                    gs_vals = log_graph_scale(model, global_step, pbar=pbar)
                    if gs_vals:
                        gs_history.append({'step': global_step, 'vals': gs_vals})

                # Pas de save en dry_run
                if not dry_run and global_step % CONFIG['save_every_steps'] == 0:
                    safe_pos = max(0, ds.pos - CONFIG['gradient_accumulation'] * CONFIG['batch_size'])
                    save_ckpt(model, opts, scheduler,
                              {'global_step': global_step, 'chunk_idx': chunk_idx,
                               'ds_pos': safe_pos, 'ds_epoch': ds.epoch},
                              CONFIG['ckpt_path'])

            pbar.update(1)

        pbar.close()
        del ds; gc.collect()

        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

    elapsed = (time.time() - t0) / 60
    print(f'\n{"="*70}\n  TRAINING TERMINÉ\n{"="*70}')
    print(f'  Steps : {global_step:,}  |  Temps : {elapsed:.1f}min')

    if dry_run:
        print('  [DRY RUN] Save checkpoint final.')
    save_ckpt(model, opts, scheduler,
              {'global_step': global_step, 'chunk_idx': len(chunks), 'ds_pos': 0},
              CONFIG['ckpt_path'])
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    with open(gs_history_path, 'w') as f:
        json.dump(gs_history, f, indent=2)
    plot_loss(history)
    print(f'  History      : {history_path}')
    print(f'  GS History   : {gs_history_path}')

    print('\nBye')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrompu')
    except Exception:
        print(traceback.format_exc())