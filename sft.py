#!/usr/bin/env python3

import os
os.environ["TORCHINDUCTOR_CACHE_DIR"]      = "./CompileCache"
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
os.makedirs("./CompileCache", exist_ok=True)

import sys
import time
import math
import json
import random
import traceback
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Optional, List

torch.set_float32_matmul_precision('high')

_root = os.path.dirname(__file__)
for _p in ['Core/Model', 'Core/Attention', 'Core/FeedForward', 'Core/TransformerBlock']:
    sys.path.append(os.path.join(_root, _p))

from GNT import GNT


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--no-compile',   action='store_true')
    p.add_argument('--compile-mode', default='default',
                   choices=['default', 'reduce-overhead', 'max-autotune'])
    p.add_argument('--max-samples',  type=int, default=None)
    p.add_argument('--batch-size',   type=int, default=None)
    p.add_argument('--grad-accum',   type=int, default=None)
    p.add_argument('--seq-len',      type=int, default=1024)
    p.add_argument('--only-phase',   choices=['A', 'B'], default=None)
    return p.parse_args()

ARGS   = get_args()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_CFG = dict(
    embed_dim           = 768,
    num_heads           = 12,
    n_kv_heads          = 4,
    num_layers          = 18,
    rel_rank            = 32,
    max_seq_len         = ARGS.seq_len,
    dropout             = 0.0,
    attn_every_n_layers = 4,
    use_qk_norm         = True,
    conv_kernel         = 4,
)

PHASE_CFG = {
    'A': {
        'name'          : 'Think + Conversationnel',
        'pretrain_ckpt' : './Model/gnt_pretrain.pt',
        'sft_ckpt'      : './Model/gnt_sft_phaseA.pt',
        'lr'            : 2e-5,
        'epochs'        : 1,
        'batch_size'    : 2,
        'grad_accum'    : 16,
        'max_grad_norm' : 1.0,
        'warmup_ratio'  : 0.03,
        'weight_decay'  : 0.01,
        'val_every'     : 100,
        'save_every'    : 500,
        'max_samples'   : 63_000,
        'loss_weights'  : {
            'think'    : 1.0,
            'code'     : 1.0,
            'output'   : 1.0,
            'response' : 1.0,
        },
        'datasets': [
            {
                'name'   : 'HuggingFaceTB/smol-smoltalk',
                'split'  : 'train',
                'weight' : 0.30,
                'type'   : 'messages',
                'keys'   : {'messages': 'messages'},
            },
            {
                'name'   : 'daily_dialog',
                'split'  : 'train',
                'weight' : 0.10,
                'type'   : 'dialog',
                'keys'   : {'dialog': 'dialog'},
            },
            {
                'name'               : 'open-r1/Mixture-of-Thoughts',
                'split'              : 'train',
                'weight'             : 0.50,
                'type'               : 'messages',
                'keys'               : {'messages': 'messages'},
                'filter_max_tokens'  : 900,
            },
            {
                'name'   : 'AI-MO/NuminaMath-CoT',
                'split'  : 'train',
                'weight' : 0.10,
                'type'   : 'qa',
                'keys'   : {'problem': 'problem', 'solution': 'solution'},
            },
            {
                'name'   : 'cais/mmlu',
                'split'  : 'auxiliary_train',
                'weight' : 0.05,
                'type'   : 'mmlu',
                'keys'   : {'question': 'question', 'choices': 'choices', 'answer': 'answer'},
            },
        ],
    },
    'B': {
        'name'          : 'Sandbox Python',
        'pretrain_ckpt' : './Model/gnt_sft_phaseA.pt',
        'sft_ckpt'      : './Model/gnt_sft_phaseB.pt',
        'lr'            : 5e-6,
        'epochs'        : 1,
        'batch_size'    : 2,
        'grad_accum'    : 16,
        'max_grad_norm' : 1.0,
        'warmup_ratio'  : 0.03,
        'weight_decay'  : 0.01,
        'val_every'     : 100,
        'save_every'    : 500,
        'max_samples'   : 40_000,
        'loss_weights'  : {
            'think'    : 1.0,
            'code'     : 1.5,
            'output'   : 0.5,
            'response' : 1.0,
        },
        'datasets': [
            {
                'name'   : 'ise-uiuc/Magicoder-Evol-Instruct-110K',
                'split'  : 'train',
                'weight' : 0.40,
                'type'   : 'code_instruct',
                'keys'   : {'instruction': 'instruction', 'response': 'response'},
            },
            {
                'name'   : 'm-a-p/CodeFeedback-Filtered-Instruction',
                'split'  : 'train',
                'weight' : 0.35,
                'type'   : 'code_instruct',
                'keys'   : {'instruction': 'query', 'response': 'answer'},
            },
            {
                'name'    : 'TokenBender/python_edu_instruct',
                'split'   : 'train',
                'weight'  : 0.25,
                'type'    : 'code_instruct',
                'keys'    : {'instruction': 'instruction', 'response': 'output'},
                'fallback': 'ise-uiuc/Magicoder-OSS-Instruct-75K',
            },
        ],
    },
}

SYSTEM_PROMPT_A = (
    "You are GNT, a helpful AI assistant. "
    "Use <think>...</think> to reason step by step before answering."
)
SYSTEM_PROMPT_B = (
    "You are GNT, a helpful AI assistant with Python sandbox capabilities. "
    "Use <think>...</think> to reason. "
    "Use <code>...</code> to write Python. "
    "Results appear in <output>...</output>."
)

print('\nTokenizer...')
_tok_dir = Path('./data/tokenizer_gnt')
if _tok_dir.exists():
    tokenizer = AutoTokenizer.from_pretrained(str(_tok_dir))
else:
    from transformers import AutoTokenizer as AT
    tokenizer = AT.from_pretrained('HuggingFaceTB/cosmo2-tokenizer')
    tokenizer.add_special_tokens({'additional_special_tokens': [
        '<think>', '</think>', '<code>', '</code>', '<output>', '</output>'
    ]})

tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = 'right'

VOCAB_SIZE = len(tokenizer)
EOS_ID     = tokenizer.eos_token_id
THINK_S    = tokenizer.convert_tokens_to_ids('<think>')
THINK_E    = tokenizer.convert_tokens_to_ids('</think>')
CODE_S     = tokenizer.convert_tokens_to_ids('<code>')
CODE_E     = tokenizer.convert_tokens_to_ids('</code>')
OUT_S      = tokenizer.convert_tokens_to_ids('<output>')
OUT_E      = tokenizer.convert_tokens_to_ids('</output>')
IM_START   = tokenizer.convert_tokens_to_ids('<|im_start|>')
IM_END     = tokenizer.convert_tokens_to_ids('<|im_end|>')
NEWLINE_ID = tokenizer.encode('\n', add_special_tokens=False)[0]

print(f'  vocab={VOCAB_SIZE}  eos={EOS_ID}')
print(f'  <think>={THINK_S}  </think>={THINK_E}')
print(f'  <code>={CODE_S}  </code>={CODE_E}')
print(f'  <output>={OUT_S}  </output>={OUT_E}')


def format_messages(messages: List[dict], system_prompt: str) -> str:
    text = f'<|im_start|>system\n{system_prompt}<|im_end|>\n'
    for msg in messages:
        role    = msg.get('role', 'user')
        content = msg.get('content', '') or ''
        text   += f'<|im_start|>{role}\n{content}<|im_end|>\n'
    return text


def format_qa(question: str, answer: str, system_prompt: str) -> str:
    return (
        f'<|im_start|>system\n{system_prompt}<|im_end|>\n'
        f'<|im_start|>user\n{question}<|im_end|>\n'
        f'<|im_start|>assistant\n{answer}<|im_end|>\n'
    )


def format_dialog(dialog: List[str], system_prompt: str) -> str:
    text  = f'<|im_start|>system\n{system_prompt}<|im_end|>\n'
    roles = ['user', 'assistant']
    for i, utt in enumerate(dialog):
        role  = roles[i % 2]
        text += f'<|im_start|>{role}\n{utt.strip()}<|im_end|>\n'
    return text


def convert_code_blocks(text: str) -> str:
    import re
    text = re.sub(
        r'```python\n(.*?)```',
        lambda m: f'<code>\n{m.group(1)}</code>',
        text, flags=re.DOTALL,
    )
    def maybe_conv(m):
        c = m.group(1)
        if any(s in c for s in ['print(', 'def ', 'import ', ' = ', 'for ', 'if ']):
            return f'<code>\n{c}</code>'
        return m.group(0)
    text = re.sub(r'```\n(.*?)```', maybe_conv, text, flags=re.DOTALL)
    text = re.sub(
        r'(?:Output|Result):\n((?:(?!(?:Output|Result):|\n\n).)+)',
        lambda m: f'<output>\n{m.group(1).rstrip()}\n</output>\n',
        text, flags=re.DOTALL | re.IGNORECASE,
    )
    return text


def compute_segment_weights(
    ids          : List[int],
    loss_weights : dict,
    phase        : str,
) -> List[float]:
    weights = [0.0] * len(ids)

    in_assistant = False
    segment      = 'response'

    i = 0
    while i < len(ids):
        tok = ids[i]

        if tok == IM_START:
            if i + 1 < len(ids):
                role_toks = []
                j = i + 1
                while j < len(ids) and ids[j] != NEWLINE_ID:
                    role_toks.append(ids[j])
                    j += 1
                role_text = tokenizer.decode(role_toks).strip()
                in_assistant = (role_text == 'assistant')
                segment      = 'response'
            i += 1
            continue

        if tok == IM_END:
            in_assistant = False
            i += 1
            continue

        if not in_assistant:
            i += 1
            continue

        # BUG-04 fix : appliquer le poids AVANT de changer le segment
        weights[i] = loss_weights.get(segment, 1.0)

        if tok == THINK_S:
            segment = 'think'
        elif tok == THINK_E:
            segment = 'response'
        elif tok == CODE_S:
            segment = 'code'
        elif tok == CODE_E:
            segment = 'response'
        elif tok == OUT_S:
            segment = 'output'
        elif tok == OUT_E:
            segment = 'response'

        i += 1

    return weights


class GNTSFTDataset(Dataset):
    def __init__(
        self,
        examples     : List[str],
        max_seq_len  : int,
        loss_weights : dict,
        phase        : str,
    ):
        self.samples     = []
        self.max_seq_len = max_seq_len
        skipped          = 0

        print(f'  Tokenisation {len(examples):,} exemples...')
        for text in tqdm(examples, desc='  Tokenize', leave=False):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < 8:
                skipped += 1
                continue

            ids = ids[:max_seq_len + 1]

            input_ids = ids[:-1]
            targets   = ids[1:]
            weights   = compute_segment_weights(input_ids, loss_weights, phase)

            if sum(w > 0 for w in weights) < 2:
                skipped += 1
                continue

            self.samples.append((
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(targets,   dtype=torch.long),
                torch.tensor(weights,   dtype=torch.float),
            ))

        print(f'  Samples : {len(self.samples):,}  |  Skipped : {skipped}')

    def __len__(self):  return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def collate_fn(batch):
    ids_list, tgt_list, w_list = zip(*batch)
    max_len = max(x.size(0) for x in ids_list)
    B = len(batch)
    # BUG-09 fix : padding avec eos_token_id au lieu de 0
    ids = torch.full((B, max_len), EOS_ID, dtype=torch.long)
    tgt = torch.full((B, max_len), -100,   dtype=torch.long)
    w   = torch.zeros(B, max_len,           dtype=torch.float)
    for i, (x, y, ww) in enumerate(zip(ids_list, tgt_list, w_list)):
        L = x.size(0)
        ids[i, :L] = x
        tgt[i, :L] = y
        w[i, :L]   = ww
    return ids, tgt, w


def load_phase_data(cfg: dict, max_samples: Optional[int], phase: str) -> List[str]:
    from datasets import load_dataset

    sys_prompt = SYSTEM_PROMPT_A if phase == 'A' else SYSTEM_PROMPT_B
    total_max  = max_samples or cfg['max_samples']
    all_texts  = []

    for ds_cfg in cfg['datasets']:
        n_target = int(total_max * ds_cfg['weight'])
        print(f'\n  → {ds_cfg["name"]}  ({ds_cfg["weight"]*100:.0f}%  ~{n_target:,} exemples)')

        try:
            ds = load_dataset(ds_cfg['name'], split=ds_cfg['split'])
        except Exception as e:
            fallback = ds_cfg.get('fallback')
            if fallback:
                print(f'    ⚠️  Fallback vers {fallback}')
                ds = load_dataset(fallback, split='train')
            else:
                print(f'    ❌ Erreur : {e} — skip')
                continue

        prefetch = 4 if ds_cfg.get('filter_max_tokens') else 2
        ds = ds.shuffle(seed=42).select(range(min(n_target * prefetch, len(ds))))

        texts = []
        for ex in ds:
            dtype = ds_cfg['type']
            try:
                if dtype == 'messages':
                    msgs = ex.get(ds_cfg['keys']['messages'], [])
                    if not msgs: continue
                    text = format_messages(msgs, sys_prompt)
                    if ds_cfg.get('filter_max_tokens'):
                        n_tok = len(tokenizer.encode(text, add_special_tokens=False))
                        if n_tok > ds_cfg['filter_max_tokens']:
                            continue

                elif dtype == 'dialog':
                    dlg = ex.get(ds_cfg['keys']['dialog'], [])
                    if not dlg or len(dlg) < 2: continue
                    text = format_dialog(dlg, sys_prompt)

                elif dtype == 'qa':
                    q = ex.get(ds_cfg['keys']['problem'], '')
                    a = ex.get(ds_cfg['keys']['solution'], '')
                    if not q or not a: continue
                    text = format_qa(q, a, sys_prompt)

                elif dtype == 'mmlu':
                    q       = ex.get(ds_cfg['keys']['question'], '')
                    choices = ex.get(ds_cfg['keys']['choices'], [])
                    ans_idx = ex.get(ds_cfg['keys']['answer'], 0)
                    if not q or not choices: continue
                    ans_idx = int(ans_idx)
                    if ans_idx < 0 or ans_idx >= len(choices): continue
                    labels  = ['A', 'B', 'C', 'D']
                    opts    = '\n'.join(f'{labels[i]}. {c}' for i, c in enumerate(choices))
                    answer  = f'{labels[ans_idx]}. {choices[ans_idx]}'
                    text    = format_qa(f'{q}\n{opts}', answer, sys_prompt)

                elif dtype == 'code_instruct':
                    inst = ex.get(ds_cfg['keys']['instruction'], '')
                    resp = ex.get(ds_cfg['keys']['response'], '')
                    if not inst or not resp: continue
                    resp = convert_code_blocks(resp)
                    text = format_qa(inst, resp, sys_prompt)

                else:
                    continue

                texts.append(text)
                if len(texts) >= n_target:
                    break

            except Exception:
                continue

        print(f'    ✅ {len(texts):,} exemples formatés')
        all_texts.extend(texts)

    return all_texts


def get_lr(step: int, total: int, cfg: dict) -> float:
    warmup = max(1, int(total * cfg['warmup_ratio']))
    min_lr = cfg['lr'] * 0.1
    if step < warmup:
        return cfg['lr'] * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return min_lr + (cfg['lr'] - min_lr) * 0.5 * (1 + math.cos(math.pi * p))


def _zeropower(G, steps=5):
    assert G.ndim >= 2
    a, b, c = 3.4445, -4.7750, 2.0315
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
                 mom_warmup_steps=100):
        # BUG-05 fix : ns_steps=5 (comme pretrain) + mom_warmup_steps
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay,
                        use_mars=use_mars, mars_gamma=mars_gamma,
                        mom_warmup_steps=mom_warmup_steps, _step=0)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, nest = group['lr'], group['nesterov']
            ns, wd   = group['ns_steps'], group['weight_decay']
            use_mars, mg = group.get('use_mars', True), group.get('mars_gamma', 0.025)
            warmup_steps = group.get('mom_warmup_steps', 0)
            base_mom     = group['momentum']
            group['_step'] = group.get('_step', 0) + 1
            t = group['_step']
            if warmup_steps > 0 and t <= warmup_steps:
                mom = 0.85 + (base_mom - 0.85) * (t / warmup_steps)
            else:
                mom = base_mom
            for p in group['params']:
                if p.grad is None or p.grad.ndim < 2: continue
                g = p.grad
                s = self.state[p]
                if use_mars:
                    if 'prev_grad' not in s: s['prev_grad'] = torch.zeros_like(g)
                    prev = s['prev_grad']
                    c_t  = torch.clamp((mg/(1-mg))*(g.norm()+1e-8)/(prev.norm()+1e-8), max=1.0)
                    g    = g + c_t * (g - prev)
                    s['prev_grad'].copy_(p.grad)
                if 'buf' not in s: s['buf'] = torch.zeros_like(g)
                buf = s['buf']
                buf.mul_(mom).add_(g)
                g   = (g + mom*buf) if nest else buf
                g   = _zeropower(g, steps=ns)
                g   = g * max(g.size(0), g.size(1)) ** 0.5
                if wd: p.mul_(1. - lr*wd)
                p.add_(g, alpha=-lr)


def configure_optimizers(model, lr, weight_decay):
    EXCLUDE = {'token_embeddings.weight', 'output_head.weight'}
    muon_p, adamw_d, adamw_n = [], [], []
    for pn, p in model.named_parameters():
        if not p.requires_grad: continue
        if pn in EXCLUDE: adamw_n.append(p); continue
        if p.ndim >= 2 and 'norm' not in pn.lower() and 'embed' not in pn.lower():
            muon_p.append(p)
        elif p.ndim >= 2: adamw_d.append(p)
        else: adamw_n.append(p)

    muon  = Muon([{'params': muon_p, 'is_muon': True}],
                 lr=lr*5, momentum=0.95, weight_decay=weight_decay,
                 ns_steps=5, mom_warmup_steps=100)
    adamw = torch.optim.AdamW(
        [{'params': adamw_d, 'weight_decay': weight_decay},
         {'params': adamw_n, 'weight_decay': 0.0}],
        lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=torch.cuda.is_available(),
    )
    print(f'  Muon+MARS : {sum(p.numel() for p in muon_p)/1e6:.2f}M  lr={lr*5:.2e}')
    print(f'  AdamW     : {(sum(p.numel() for p in adamw_d)+sum(p.numel() for p in adamw_n))/1e6:.2f}M  lr={lr:.2e}')
    return muon, adamw


@torch.no_grad()
def evaluate(model, val_loader, phase_cfg) -> float:
    model.eval()
    total, n = 0.0, 0
    ae  = DEVICE == 'cuda'
    adt = torch.bfloat16 if ae else torch.float32
    for ids, tgt, w in val_loader:
        ids, tgt, w = ids.to(DEVICE), tgt.to(DEVICE), w.to(DEVICE)
        with torch.amp.autocast(DEVICE, dtype=adt, enabled=ae):
            logits, _, _ = model(ids)
            loss_raw = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                tgt.reshape(-1),
                ignore_index=-100,
                reduction='none',
            )
            w_flat = w.reshape(-1)
            n_tok  = w_flat.sum().clamp(min=1)
            loss   = (loss_raw * w_flat).sum() / n_tok
        total += loss.item(); n += 1
        if n >= 50: break
    model.train()
    return total / max(n, 1)


def save(model, optimizers, step, val_loss, path, periodic=False):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    m = model._orig_mod if hasattr(model, '_orig_mod') else model
    muon, adamw = optimizers
    # BUG-15 fix : distinguer save périodique du save best via la clé 'periodic'
    torch.save({
        'model_state_dict'  : m.state_dict(),
        'muon_state_dict'   : muon.state_dict(),
        'adamw_state_dict'  : adamw.state_dict(),
        'step'              : step,
        'val_loss'          : val_loss,
        'periodic'          : periodic,
    }, path + '.tmp')
    os.replace(path + '.tmp', path)
    tag = '[periodic]' if periodic else '[best]'
    print(f'\n  💾 SAVE {tag} step={step:,}  val_loss={val_loss:.4f}  [{path}]')


def run_phase(phase: str):
    cfg = PHASE_CFG[phase]

    if ARGS.batch_size is not None:
        cfg['batch_size'] = ARGS.batch_size
    if ARGS.grad_accum is not None:
        cfg['grad_accum'] = ARGS.grad_accum

    print()
    print('=' * 70)
    print(f'  GNT v1 — SFT Phase {phase} : {cfg["name"]}')
    print('=' * 70)
    print(f'  Ckpt in  : {cfg["pretrain_ckpt"]}')
    print(f'  Ckpt out : {cfg["sft_ckpt"]}')
    print(f'  LR       : {cfg["lr"]}  |  seq={MODEL_CFG["max_seq_len"]}')
    print(f'  Device   : {DEVICE}')
    if DEVICE == 'cuda':
        print(f'  GPU      : {torch.cuda.get_device_name(0)}')
        print(f'  VRAM     : {torch.cuda.get_device_properties(0).total_memory//1024**3} GB')
    print(f'  Loss weights : {cfg["loss_weights"]}')
    print('=' * 70)

    print('\nChargement datasets...')
    all_texts = load_phase_data(cfg, ARGS.max_samples, phase)
    print(f'\n  Total : {len(all_texts):,} exemples')

    n_val   = max(100, int(len(all_texts) * 0.02))
    n_train = len(all_texts) - n_val

    # BUG-16 fix : Random local pour ne pas polluer le rng global
    _rng = random.Random(42)
    _rng.shuffle(all_texts)

    train_texts = all_texts[:n_train]
    val_texts   = all_texts[n_train:]

    train_ds = GNTSFTDataset(train_texts, MODEL_CFG['max_seq_len'],
                              cfg['loss_weights'], phase)
    val_ds   = GNTSFTDataset(val_texts,   MODEL_CFG['max_seq_len'],
                              cfg['loss_weights'], phase)

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size']*2,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=2, pin_memory=True)

    print('\nChargement modèle...')
    model = GNT(vocab_size=VOCAB_SIZE, **MODEL_CFG).to(DEVICE)

    ckpt = torch.load(cfg['pretrain_ckpt'], map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    missing, _ = model.load_state_dict(state, strict=False)
    print(f'  Weights OK  |  missing={len(missing)}')

    p = model.count_parameters()
    print(f'  Params : {p["total_M"]}M  |  Naylis : {p["naylis_K"]}K ({p["naylis_pct"]})')

    model = model.to(torch.bfloat16)

    if not ARGS.no_compile and DEVICE == 'cuda':
        print(f'\ntorch.compile (mode={ARGS.compile_mode})...')
        import torch._dynamo as _d
        _d.config.cache_size_limit = 256
        _d.config.suppress_errors  = True
        model = torch.compile(model, mode=ARGS.compile_mode)
        print('  OK')

    raw  = model._orig_mod if hasattr(model, '_orig_mod') else model
    opts = configure_optimizers(raw, cfg['lr'], cfg['weight_decay'])
    muon, adamw = opts

    total_steps = (len(train_loader) // cfg['grad_accum']) * cfg['epochs']
    print(f'\n  Total steps    : {total_steps:,}')
    print(f'  Batch effectif : {cfg["batch_size"] * cfg["grad_accum"]}')

    print('\n' + '=' * 70)
    print(f'  SFT Phase {phase} START')
    print('=' * 70 + '\n')

    model.train()
    ae  = DEVICE == 'cuda'
    adt = torch.bfloat16 if ae else torch.float32

    global_step   = 0
    acc_steps     = 0
    acc_loss      = 0.0
    best_val_loss = float('inf')
    history       = []
    t0            = time.time()

    muon.zero_grad(set_to_none=True)
    adamw.zero_grad(set_to_none=True)

    pbar = tqdm(train_loader, desc=f'SFT-{phase}', dynamic_ncols=True, colour='green')

    for ids, tgt, w in pbar:
        ids, tgt, w = ids.to(DEVICE), tgt.to(DEVICE), w.to(DEVICE)

        with torch.amp.autocast(DEVICE, dtype=adt, enabled=ae):
            logits, _, _ = model(ids)
            loss_raw = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                tgt.reshape(-1),
                ignore_index=-100,
                reduction='none',
            )
            w_flat = w.reshape(-1)
            n_tok  = w_flat.sum().clamp(min=1)
            loss   = (loss_raw * w_flat).sum() / n_tok
            loss   = loss / cfg['grad_accum']

        loss.backward()
        acc_loss  += loss.item()
        acc_steps += 1

        if acc_steps >= cfg['grad_accum']:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['max_grad_norm'])

            lr = get_lr(global_step, total_steps, cfg)
            for pg in muon.param_groups:  pg['lr'] = lr * 5
            for pg in adamw.param_groups: pg['lr'] = lr

            muon.step()
            adamw.step()

            muon.zero_grad(set_to_none=True)
            adamw.zero_grad(set_to_none=True)

            global_step += 1
            tl  = acc_loss
            ppl = math.exp(min(tl, 20))
            acc_loss  = 0.0
            acc_steps = 0

            pbar.set_postfix(
                loss=f'{tl:.4f}', ppl=f'{ppl:.1f}',
                lr=f'{lr:.2e}', step=global_step,
            )

            if global_step % cfg['val_every'] == 0:
                vl  = evaluate(model, val_loader, cfg)
                vpp = math.exp(min(vl, 20))
                elapsed = (time.time() - t0) / 60
                pbar.write(
                    f'  [val step={global_step:,}] '
                    f'loss={vl:.4f}  ppl={vpp:.2f}  {elapsed:.1f}min'
                )

                m = model._orig_mod if hasattr(model, '_orig_mod') else model
                gs = [b.attention.graph_scale.abs().mean().item()
                      for b in m.blocks if hasattr(b, 'attention')
                      and hasattr(b.attention, 'graph_scale')]
                if gs:
                    pbar.write(
                        f'  [naylis step={global_step:,}] '
                        f'|graph_scale| avg={sum(gs)/len(gs):.5f}  max={max(gs):.5f}'
                    )

                history.append({
                    'step': global_step, 'phase': phase,
                    'val_loss': vl, 'val_ppl': vpp, 'lr': lr,
                })

                if vl < best_val_loss:
                    best_val_loss = vl
                    save(model, opts, global_step, vl, cfg['sft_ckpt'], periodic=False)

            elif global_step % cfg['save_every'] == 0:
                save(model, opts, global_step, float('nan'), cfg['sft_ckpt'], periodic=True)

    elapsed = (time.time() - t0) / 60
    vl = evaluate(model, val_loader, cfg)
    print(f'\n{"="*70}\n  SFT Phase {phase} TERMINÉ\n{"="*70}')
    print(f'  Steps    : {global_step:,}  |  Temps : {elapsed:.1f}min')
    print(f'  Val loss : {vl:.4f}  |  Best  : {best_val_loss:.4f}')
    save(model, opts, global_step, vl, cfg['sft_ckpt'], periodic=False)

    hist_path = cfg['sft_ckpt'].replace('.pt', '_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f'  History  : {hist_path}')

    del model, train_ds, val_ds, train_loader, val_loader
    import gc; gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()


def main():
    phases = ['A', 'B'] if ARGS.only_phase is None else [ARGS.only_phase]
    for phase in phases:
        run_phase(phase)
    print('\nBye')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrompu')
    except Exception:
        print(traceback.format_exc())