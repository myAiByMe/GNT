#!/usr/bin/env python3
"""
benchmark_gnt.py — GNT v1
==========================
Deux modes d'évaluation via lm-evaluation-harness (EleutherAI) :

  MODE PRETRAIN (défaut) — log-likelihood, comparable aux papiers :
    HellaSwag / ARC-Easy / ARC-Challenge(25-shot) / WinoGrande / PIQA
    MMLU(5-shot) / TriviaQA(0-shot)

  MODE SFT — même benchmarks pretrain + benchmarks instruct :
    IFEval (instruction following strict, 0-shot)
    GSM8K  (raisonnement mathématique CoT, 5-shot)
    TruthfulQA MC1 (éviter les mythes, 0-shot)
    Le wrapper SFT encapsule chaque prompt en ChatML GNT avec
    <|im_start|>system / user / assistant pour que le modèle
    réponde dans le bon format (think, code, output inclus).

USAGE autonome :
  python3.10 benchmark_gnt.py --ckpt ./Model/gnt_pretrain.pt
  python3.10 benchmark_gnt.py --ckpt ./Model/gnt_sft_phaseB.pt --mode sft --label B

  # Lancer seulement TriviaQA :
  python3.10 benchmark_gnt.py --ckpt ./Model/gnt_pretrain.pt --benchmarks triviaqa

  # Lancer ARC-Easy + HellaSwag + MMLU :
  python3.10 benchmark_gnt.py --ckpt ./Model/gnt_pretrain.pt --benchmarks arc_easy hellaswag mmlu

  # Lancer tous les benchmarks SFT sauf GSM8K :
  python3.10 benchmark_gnt.py --ckpt ./Model/gnt_sft.pt --mode sft --exclude gsm8k

  Benchmarks disponibles :
    pretrain : hellaswag arc_easy arc_challenge winogrande piqa mmlu triviaqa
    ablation : nq_open webqs fever
    sft extra : ifeval gsm8k truthfulqa

USAGE depuis sft_gnt.py :
  from benchmark_gnt import run_benchmarks
  run_benchmarks(model, tokenizer, device, label='A', out_dir='./Model', mode='sft')
  run_benchmarks(model, tokenizer, device, only=['triviaqa', 'mmlu'])
"""

import os
import gc
import json
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List

from lm_eval.api.model import LM
from lm_eval import simple_evaluate


# ── Prompts système GNT ───────────────────────────────────────────────────────
SYSTEM_SFT_A = (
    "You are GNT, a helpful AI assistant. "
    "Use <think>...</think> to reason step by step before answering."
)
SYSTEM_SFT_B = (
    "You are GNT, a helpful AI assistant with Python sandbox capabilities. "
    "Use <think>...</think> to reason. "
    "Use <code>...</code> to write Python. "
    "Results appear in <o>...</o>."
)


def _chatml(system: str, user: str) -> str:
    """Formate un prompt en ChatML GNT."""
    out = ''
    if system:
        out += f'<|im_start|>system\n{system}<|im_end|>\n'
    out += f'<|im_start|>user\n{user}<|im_end|>\n'
    out += '<|im_start|>assistant\n'
    return out


# ── Définition des benchmarks ─────────────────────────────────────────────────
PRETRAIN_TASK_MAP = {
    'hellaswag'    : ('hellaswag',      0),
    'arc_easy'     : ('arc_easy',       0),
    'arc_challenge': ('arc_challenge',  25),  # 25-shot standard
    'winogrande'   : ('winogrande',     0),
    'piqa'         : ('piqa',           0),
    'mmlu'         : ('mmlu',           5),
    'triviaqa'     : ('triviaqa',       0),
    # ── Ablation NaylisAttention — mémoire relationnelle / factuelle ──────────
    'nq_open'      : ('nq_open',        0),   # NaturalQuestions open-domain
    'webqs'        : ('webqs',          0),   # WebQuestions — relations entités
    'fever'        : ('fever',          0),   # FEVER — fact verification
}

SFT_EXTRA_TASK_MAP = {
    'ifeval'    : ('ifeval',        0),
    'gsm8k'     : ('gsm8k',         5),
    'truthfulqa': ('truthfulqa_mc1', 0),
}

BENCH_LABELS = {
    'hellaswag'    : 'HellaSwag',
    'arc_easy'     : 'ARC-Easy',
    'arc_challenge': 'ARC-Challenge',
    'winogrande'   : 'WinoGrande',
    'piqa'         : 'PIQA',
    'mmlu'         : 'MMLU',
    'triviaqa'     : 'TriviaQA',
    'nq_open'      : 'NaturalQuestions',
    'webqs'        : 'WebQuestions',
    'fever'        : 'FEVER',
    'ifeval'       : 'IFEval',
    'gsm8k'        : 'GSM8K',
    'truthfulqa'   : 'TruthfulQA',
}

RANDOM_BASELINES = {
    'hellaswag'    : 0.25,
    'arc_easy'     : 0.25,
    'arc_challenge': 0.25,
    'winogrande'   : 0.50,
    'piqa'         : 0.50,
    'mmlu'         : 0.25,
    'triviaqa'     : 0.00,
    'nq_open'      : 0.00,
    'webqs'        : 0.00,
    'fever'        : 0.50,   # classification 3 classes : SUPPORTS/REFUTES/NEI
    'ifeval'       : 0.00,
    'gsm8k'        : 0.00,
    'truthfulqa'   : 0.25,
}

BENCH_COLORS = {
    'hellaswag'    : '#4fc3f7',
    'arc_easy'     : '#81c784',
    'arc_challenge': '#ffb74d',
    'winogrande'   : '#ce93d8',
    'piqa'         : '#80cbc4',
    'mmlu'         : '#f48fb1',
    'triviaqa'     : '#ffcc80',
    'nq_open'      : '#80deea',
    'webqs'        : '#b39ddb',
    'fever'        : '#ff8a65',
    'ifeval'       : '#ef9a9a',
    'gsm8k'        : '#a5d6a7',
    'truthfulqa'   : '#b0bec5',
}

PRETRAIN_BENCHMARKS = list(PRETRAIN_TASK_MAP.keys())
SFT_BENCHMARKS      = PRETRAIN_BENCHMARKS + list(SFT_EXTRA_TASK_MAP.keys())
ALL_TASK_MAP        = {**PRETRAIN_TASK_MAP, **SFT_EXTRA_TASK_MAP}


# ── Wrapper GNT pour lm-eval ──────────────────────────────────────────────────
class GNTLMWrapper(LM):
    """
    Wrapper lm-eval pour GNT.
    mode='pretrain' : prompts bruts, pas de ChatML.
    mode='sft'      : prompts encapsulés en ChatML GNT avec system prompt.
    """

    def __init__(self, model, tokenizer, device: str,
                 batch_size: int = 4, max_seq_len: int = 1024,
                 mode: str = 'pretrain', system_prompt: str = ''):
        super().__init__()
        self.model           = model
        self.tokenizer       = tokenizer
        self.device          = device
        self._batch_size_val = batch_size
        self.max_seq_len     = max_seq_len
        self._dtype          = torch.bfloat16 if device == 'cuda' else torch.float32
        self.mode            = mode
        self.system_prompt   = system_prompt

    # ── Toutes les propriétés/méthodes requises par lm-eval 0.4.x ───────────
    @property
    def world_size(self) -> int:
        return 1

    @property
    def rank(self) -> int:
        return 0

    @property
    def accelerator(self):
        return None

    @property
    def tokenizer_name(self) -> str:
        return getattr(self.tokenizer, 'name_or_path', 'gnt-tokenizer')

    def apply_chat_template(self, chat_history: list) -> str:
        """Formate un historique ChatML pour lm-eval (mode SFT)."""
        if self.mode == 'sft':
            system = next(
                (m['content'] for m in chat_history if m['role'] == 'system'),
                self.system_prompt,
            )
            user = next(
                (m['content'] for m in chat_history if m['role'] == 'user'),
                '',
            )
            return _chatml(system, user)
        return ' '.join(m.get('content', '') for m in chat_history)

    @property
    def chat_template(self) -> str:
        return ''

    # ── Propriétés standard ──────────────────────────────────────────────────

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self.max_seq_len

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size_val

    def tok_encode(self, text: str) -> list:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def tok_decode(self, tokens) -> str:
        return self.tokenizer.decode(tokens)

    def _wrap_context(self, context: str) -> str:
        if self.mode == 'sft' and context:
            return _chatml(self.system_prompt, context)
        return context

    def _encode_pair(self, context: str, continuation: str):
        ctx_ids = self.tok_encode(self._wrap_context(context)) if context else []
        con_ids = self.tok_encode(continuation)
        if not con_ids:
            con_ids = self.tok_encode(' ' + continuation)
        full = ctx_ids + con_ids
        if len(full) > self.max_seq_len:
            full    = full[-self.max_seq_len:]
            ctx_len = max(1, len(full) - len(con_ids))
        else:
            ctx_len = len(ctx_ids)
        return full, ctx_len, len(con_ids)

    @torch.no_grad()
    def loglikelihood(self, requests: list) -> list:
        results = []
        pad_id  = self.eot_token_id or 0
        from tqdm import tqdm

        batches = range(0, len(requests), self._batch_size_val)
        for i in tqdm(batches, desc='  loglikelihood', unit='batch',
                      dynamic_ncols=True, leave=False):
            batch_reqs = requests[i : i + self._batch_size_val]
            batch_data = [self._encode_pair(*req.args) for req in batch_reqs]

            max_len   = max(len(d[0]) for d in batch_data)
            input_ids = torch.full(
                (len(batch_data), max_len), pad_id,
                dtype=torch.long, device=self.device,
            )
            for j, (full_ids, _, _) in enumerate(batch_data):
                input_ids[j, :len(full_ids)] = torch.tensor(
                    full_ids, dtype=torch.long, device=self.device)

            with torch.amp.autocast(self.device, dtype=self._dtype,
                                    enabled=(self.device == 'cuda')):
                logits, _, _ = self.model(input_ids)

            log_probs = F.log_softmax(logits, dim=-1)

            for j, (full_ids, ctx_len, con_len) in enumerate(batch_data):
                start    = ctx_len - 1
                end      = min(ctx_len + con_len - 1, log_probs.shape[1])
                lp_slice = log_probs[j, start:end, :]
                tgt      = torch.tensor(
                    full_ids[ctx_len : ctx_len + con_len],
                    dtype=torch.long, device=self.device,
                )[:lp_slice.shape[0]]

                token_lp = lp_slice[range(len(tgt)), tgt]
                logprob  = token_lp.sum().item()
                greedy   = (lp_slice.argmax(dim=-1) == tgt).all().item()
                results.append((logprob, bool(greedy)))
        return results

    @torch.no_grad()
    def loglikelihood_rolling(self, requests: list) -> list:
        results = []
        for req in requests:
            (text,)   = req.args
            token_ids = self.tok_encode(text)
            if not token_ids:
                results.append(0.0)
                continue

            total_lp = 0.0
            stride   = self.max_seq_len

            for start in range(0, len(token_ids), stride):
                chunk = token_ids[max(0, start - 1) : start + stride]
                ids_t = torch.tensor([chunk], dtype=torch.long, device=self.device)
                x, y  = ids_t[:, :-1], ids_t[:, 1:]
                with torch.amp.autocast(self.device, dtype=self._dtype,
                                        enabled=(self.device == 'cuda')):
                    logits, _, _ = self.model(x)
                lp         = F.log_softmax(logits, dim=-1)
                score_from = 1 if start > 0 else 0
                lp_tgt     = lp[0, score_from:].gather(
                    -1, y[0, score_from:].unsqueeze(-1)
                ).squeeze(-1)
                total_lp += lp_tgt.sum().item()

            results.append(total_lp)
        return results

    @torch.no_grad()
    def generate_until(self, requests: list) -> list:
        """
        En mode SFT, encapsule le contexte en ChatML avant génération.

        FIX : les stops 'until' (ex: '\n', '.', ',') sont convertis en token IDs
        et passés directement à model.generate() via stop_token_ids.
        Sans ça, le modèle générait max_gen_toks=256 tokens pour chaque question
        au lieu de s'arrêter après 3-5 tokens → freeze sur 17 944 samples TriviaQA.
        """
        results = []
        from tqdm import tqdm

        # EOS token — vérifier que im_end_id est valide et pas UNK
        im_end_id = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        unk_id    = getattr(self.tokenizer, 'unk_token_id', None)
        if im_end_id == unk_id:
            im_end_id = None
        eos_id = im_end_id if (self.mode == 'sft' and im_end_id is not None) \
                 else self.eot_token_id

        for req in tqdm(requests, desc='  generate_until', unit='q', dynamic_ncols=True):
            context, gen_kwargs = req.args
            until    = gen_kwargs.get('until', [self.tokenizer.eos_token])
            max_toks = gen_kwargs.get('max_gen_toks', self.max_gen_toks)

            prompt    = self._wrap_context(context)
            token_ids = self.tok_encode(prompt)

            # Truncate par la droite pour préserver la fin de la question
            if len(token_ids) > self.max_seq_len - max_toks:
                token_ids = token_ids[-(self.max_seq_len - max_toks):]

            input_ids = torch.tensor(
                [token_ids], dtype=torch.long, device=self.device
            )

            # Convertir les until strings en token IDs (seulement les stops d'1 token)
            # C'est le fix principal : sans ça TriviaQA génère 256 tokens par question
            stop_token_ids = []
            for s in until:
                if not s:
                    continue
                ids = self.tok_encode(s)
                if len(ids) == 1:
                    stop_token_ids.append(ids[0])
                # Si le stop est multi-token (rare), on laisse le post-processing gérer

            output_ids = self.model.generate(
                input_ids,
                max_new_tokens = max_toks,
                temperature    = 0.0,
                eos_token_id   = eos_id,
                stop_token_ids = stop_token_ids if stop_token_ids else None,
            )
            gen_tokens = output_ids[0, input_ids.shape[1]:]
            generated  = self.tok_decode(gen_tokens.tolist())

            # Post-processing : stops multi-tokens + im_end en mode SFT
            all_stops = list(until)
            if self.mode == 'sft':
                all_stops.append('<|im_end|>')
            for stop in all_stops:
                if stop and stop in generated:
                    generated = generated[:generated.index(stop)]

            results.append(generated.strip())
        return results


# ── Plot ──────────────────────────────────────────────────────────────────────
def _style_ax(ax):
    ax.set_facecolor('#1a1d27')
    ax.tick_params(colors='#cccccc')
    ax.xaxis.label.set_color('#cccccc')
    ax.yaxis.label.set_color('#cccccc')
    ax.title.set_color('#ffffff')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333344')
    ax.grid(True, color='#2a2d3a', linewidth=0.5)


def plot_benchmarks(bench_history: list, path: str, mode: str = 'pretrain'):
    if not bench_history:
        return

    bench_list = SFT_BENCHMARKS if mode == 'sft' else PRETRAIN_BENCHMARKS
    labels     = [h.get('label', str(i)) for i, h in enumerate(bench_history)]

    fig, axes = plt.subplots(
        len(bench_list), 1,
        figsize=(12, 3.5 * len(bench_list)),
        sharex=True,
    )
    if len(bench_list) == 1:
        axes = [axes]
    fig.patch.set_facecolor('#0e1117')

    for ax, bench in zip(axes, bench_list):
        _style_ax(ax)
        scores = [h.get(bench) for h in bench_history]
        valid  = [(l, s) for l, s in zip(labels, scores) if s is not None]
        if not valid:
            ax.set_title(BENCH_LABELS.get(bench, bench), pad=8)
            continue

        lx, sy   = zip(*valid)
        color    = BENCH_COLORS.get(bench, '#ffffff')
        baseline = RANDOM_BASELINES.get(bench, 0.0)
        x_pos    = list(range(len(lx)))

        if baseline > 0:
            ax.axhline(baseline, color='#555566', linewidth=0.8,
                       linestyle='--', label=f'Random ({baseline*100:.0f}%)')
        ax.plot(x_pos, sy, color=color, linewidth=2.0,
                marker='o', markersize=7, label=BENCH_LABELS.get(bench, bench))
        for xp, yp in zip(x_pos, sy):
            ax.annotate(f'{yp*100:.1f}%', xy=(xp, yp), xytext=(0, 8),
                        textcoords='offset points',
                        ha='center', color=color, fontsize=8)

        ax.set_ylabel('Accuracy', color='#cccccc')
        ax.set_title(BENCH_LABELS.get(bench, bench), pad=8)
        ax.set_ylim(0, 1.08)
        ax.legend(facecolor='#1a1d27', labelcolor='#cccccc',
                  fontsize=8, loc='lower right')

    axes[-1].set_xlabel('Phase' if mode == 'sft' else 'Chunk', color='#cccccc')
    axes[-1].set_xticks(range(len(labels)))
    axes[-1].set_xticklabels(labels, rotation=20, ha='right', color='#cccccc')

    title = ('GNT SFT — Benchmarks par phase' if mode == 'sft'
             else 'GNT Pretrain — Benchmarks par chunk')
    fig.suptitle(title + ' (lm-eval-harness)',
                 color='#ffffff', fontsize=13, y=1.005)
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  📊 Benchmark graph → {path}')


# ── Fonction principale ───────────────────────────────────────────────────────
def run_benchmarks(
    model,
    tokenizer,
    device        : str,
    chunk_idx     : int            = 0,
    label         : str            = '',
    out_dir       : str            = './Model',
    batch_size    : int            = 4,
    mode          : str            = 'pretrain',
    system_prompt : str            = '',
    only          : Optional[List[str]] = None,   # ex: ['triviaqa', 'mmlu']
    exclude       : Optional[List[str]] = None,   # ex: ['gsm8k']
) -> dict:
    """
    Lance les benchmarks via lm-evaluation-harness.
    mode='pretrain' : benchmarks base model, indexés par chunk_idx.
    mode='sft'      : benchmarks pretrain + SFT instruct, indexés par label.

    only    : liste de bench_keys à lancer (ignore les autres).
    exclude : liste de bench_keys à sauter.
    """
    entry_key = label if (mode == 'sft' and label) else f'chunk_{chunk_idx:03d}'

    print('\n' + '=' * 70)
    print(f'  BENCHMARKS ({mode.upper()}) — {entry_key}')
    print('=' * 70)

    model.eval()
    raw         = model._orig_mod if hasattr(model, '_orig_mod') else model
    max_seq_len = getattr(raw, 'max_seq_len', 1024)

    # System prompt SFT automatique si non fourni
    if mode == 'sft' and not system_prompt:
        system_prompt = SYSTEM_SFT_B if label == 'B' else SYSTEM_SFT_A

    # Construire la task_map selon le mode
    task_map = dict(PRETRAIN_TASK_MAP)
    if mode == 'sft':
        task_map.update(SFT_EXTRA_TASK_MAP)

    # Filtrer selon only / exclude
    if only:
        unknown = [k for k in only if k not in ALL_TASK_MAP]
        if unknown:
            print(f'  ⚠️  Benchmarks inconnus ignorés : {unknown}')
            print(f'       Disponibles : {list(ALL_TASK_MAP.keys())}')
        task_map = {k: v for k, v in task_map.items() if k in only}
        if not task_map:
            print('  ⚠️  Aucun benchmark valide après filtrage --benchmarks.')
            return {}
        print(f'  → Benchmarks sélectionnés : {list(task_map.keys())}')

    if exclude:
        task_map = {k: v for k, v in task_map.items() if k not in exclude}
        print(f'  → Benchmarks exclus : {exclude}')

    scores = {}

    for bench_key, (task_name, n_shot) in task_map.items():
        print(f'\n  ▶ {BENCH_LABELS.get(bench_key, bench_key)}  ({n_shot}-shot)...')
        try:
            # Nouveau wrapper par benchmark pour éviter l'accumulation d'état
            _wrapper = GNTLMWrapper(
                model         = raw,
                tokenizer     = tokenizer,
                device        = device,
                batch_size    = batch_size,
                max_seq_len   = max_seq_len,
                mode          = mode,
                system_prompt = system_prompt,
            )
            results = simple_evaluate(
                model       = _wrapper,
                tasks       = [task_name],
                num_fewshot = n_shot,
                batch_size  = batch_size,
                # device= intentionnellement omis : le wrapper GNTLMWrapper
                # gère déjà le placement sur device. Le passer ici en double
                # crée un conflit interne dans lm-eval qui provoque un freeze.
                limit       = None,
                log_samples = False,
                verbosity   = 'INFO',
            )
            task_res = results['results'].get(task_name, {})
            acc = (
                task_res.get('acc_norm,none')
                or task_res.get('acc,none')
                or task_res.get('exact_match,remove_whitespace') 
                or task_res.get('exact_match,none')          # TriviaQA, NQ, WebQ, GSM8K
                or task_res.get('exact_match')
                or task_res.get('acc_norm')
                or task_res.get('acc')
                or task_res.get('prompt_level_strict_acc,none')  # IFEval strict
            )
            if acc is not None:
                scores[bench_key] = float(acc)
                print(f'    → {float(acc)*100:.2f}%')
            else:
                print(f'    ⚠️  Clé accuracy non trouvée : {list(task_res.keys())}')
        except Exception as e:
            import traceback as tb
            print(f'    ⚠️  {BENCH_LABELS.get(bench_key, bench_key)} échoué : {e}')
            tb.print_exc()

        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

    model.train()

    # Résumé console
    bench_list = list(task_map.keys())
    print('\n  ┌─ Résultats ──────────────────────────────────┐')
    for bench_key in bench_list:
        acc = scores.get(bench_key)
        if acc is not None:
            bar   = '█' * int(acc * 20)
            delta = acc - RANDOM_BASELINES.get(bench_key, 0.0)
            sign  = '+' if delta >= 0 else ''
            print(f'  │  {BENCH_LABELS.get(bench_key, bench_key):<15} {acc*100:5.1f}%  '
                  f'({sign}{delta*100:.1f}% vs random)  {bar}')
        else:
            print(f'  │  {BENCH_LABELS.get(bench_key, bench_key):<15} N/A')
    print('  └──────────────────────────────────────────────┘')

    # JSON — fichiers séparés pretrain / sft
    suffix       = 'sft' if mode == 'sft' else 'pretrain'
    history_path = os.path.join(out_dir, f'gnt_bench_{suffix}_history.json')
    bench_history = []
    if os.path.exists(history_path):
        with open(history_path) as f:
            bench_history = json.load(f)
    bench_history = [h for h in bench_history if h.get('label') != entry_key]
    bench_history.append({'label': entry_key, **scores})
    bench_history.sort(key=lambda h: h['label'])
    with open(history_path, 'w') as f:
        json.dump(bench_history, f, indent=2)
    print(f'  💾 Bench history → {history_path}')

    graph_path = os.path.join(out_dir, f'gnt_benchmarks_{suffix}.png')
    plot_benchmarks(bench_history, graph_path, mode=mode)

    return scores


# ── Usage autonome ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse, sys

    _root = os.path.dirname(__file__)
    for sub in ('Model', 'Attention', 'FeedForward', 'TransformerBlock'):
        sys.path.append(os.path.join(_root, 'Core', sub))
    from GNT import GNT
    from transformers import AutoTokenizer

    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument('--ckpt',          default='./Model/gnt_pretrain.pt')
    p.add_argument('--tokenizer-dir', default='./data/tokenizer_gnt')
    p.add_argument('--chunk-idx',     type=int, default=0)
    p.add_argument('--label',         default='',
                   help='Label pour le mode SFT (ex: A, B)')
    p.add_argument('--out-dir',       default='./Model')
    p.add_argument('--batch-size',    type=int, default=4)
    p.add_argument('--mode',          default='pretrain',
                   choices=['pretrain', 'sft'],
                   help='pretrain (défaut) ou sft')
    p.add_argument('--device',        default=None,
                   help='cuda ou cpu (défaut : auto-détection)')
    p.add_argument('--benchmarks',    nargs='+', default=None,
                   metavar='BENCH',
                   help=(
                       'Lancer seulement ces benchmarks. '
                       'Ex: --benchmarks triviaqa  ou  --benchmarks arc_easy mmlu hellaswag. '
                       f'Choix : {", ".join(ALL_TASK_MAP.keys())}'
                   ))
    p.add_argument('--exclude',       nargs='+', default=None,
                   metavar='BENCH',
                   help=(
                       'Exclure ces benchmarks. '
                       'Ex: --exclude gsm8k ifeval'
                   ))
    args = p.parse_args()

    if args.device is not None:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'  Device : {device}')
    tok_dir   = Path(args.tokenizer_dir)
    tokenizer = (AutoTokenizer.from_pretrained(str(tok_dir)) if tok_dir.exists()
                 else AutoTokenizer.from_pretrained('HuggingFaceTB/cosmo2-tokenizer'))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cp    = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg   = cp.get('config', {})
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    model = GNT(
        vocab_size          = cfg.get('vocab_size', len(tokenizer)),
        embed_dim           = cfg.get('embed_dim', 768),
        num_heads           = cfg.get('num_heads', 12),
        n_kv_heads          = cfg.get('n_kv_heads', 4),
        num_layers          = cfg.get('num_layers', 18),
        rel_rank            = cfg.get('rel_rank', 32),
        max_seq_len         = cfg.get('max_seq_len', 1024),
        dropout             = 0.0,
        attn_every_n_layers = cfg.get('attn_every_n_layers', 4),
        use_qk_norm         = cfg.get('use_qk_norm', True),
        conv_kernel         = cfg.get('conv_kernel', 4),
    ).to(dtype).to(device)

    state = cp.get('model_state_dict', cp)
    state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    print(f'  Modèle : {sum(p.numel() for p in model.parameters())/1e6:.1f}M params')

    run_benchmarks(
        model      = model,
        tokenizer  = tokenizer,
        device     = device,
        chunk_idx  = args.chunk_idx,
        label      = args.label or f'chunk_{args.chunk_idx:03d}',
        out_dir    = args.out_dir,
        batch_size = args.batch_size,
        mode       = args.mode,
        only       = args.benchmarks,
        exclude    = args.exclude,
    )