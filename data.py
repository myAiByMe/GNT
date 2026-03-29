#!/usr/bin/env python3
"""
dataset_gnt.py — GNT v1
========================
Curriculum 3 phases × 5B tokens + dry_chunk 1B = 6B tokens total.
Tokenizer : HuggingFaceTB/cosmo2-tokenizer + 6 tokens spéciaux

TOKENS SPÉCIAUX :
  <think>   → début raisonnement interne (CoT)
  </think>  → fin raisonnement interne
  <code>    → début sandbox Python
  </code>   → fin sandbox Python
  <output>    → résultat d'exécution sandbox
  </output>   → fin résultat
  vocab final : 49,158 tokens

CHUNKS (6 au total) :
  dry_chunk — 1B tokens — Cosmopedia v2 uniquement
  chunk_000 — 1B tokens — Phase 1 chunk 1
  chunk_001 — 1B tokens — Phase 1 chunk 2
  chunk_002 — 1B tokens — Phase 2 chunk 1
  chunk_003 — 1B tokens — Phase 2 chunk 2
  chunk_004 — 1B tokens — Phase 3 chunk 1

USAGE :
  python dataset_gnt.py           # télécharge tout (dry_chunk + 3 phases)
  python dataset_gnt.py --phase 1
  python dataset_gnt.py --phase 2
  python dataset_gnt.py --phase 3
"""

import os
import sys
import gc
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import List, Optional

# ── Tokens spéciaux GNT ──────────────────────────────────────────────────────
SPECIAL_TOKENS = [
    '<think>',    # début raisonnement interne
    '</think>',   # fin raisonnement interne
    '<code>',     # début sandbox Python
    '</code>',    # fin sandbox Python
    '<output>',   # résultat d'exécution sandbox
    '</output>',  # fin résultat
]

# ── Dry chunk — 1B tokens Cosmopedia v2 (smoke-test) ─────────────────────────
DRY_CHUNK = {
    'name'    : 'dry_chunk',
    'label'   : 'Dry run — 1B tokens Cosmopedia v2',
    'datasets': [
        {
            'name'            : 'cosmopedia_v2',
            'source'          : 'HuggingFaceTB/cosmopedia-v2',
            'config'          : 'cosmopedia-v2',
            'split'           : 'train',
            'text_key'        : 'text',
            'tokens_per_chunk': 1_000_000_000,  # 100% — 1B tokens
            'lang_filter'     : 'none',
            'skip_filter'     : True,
        },
    ],
}

# ── Curriculum 3 phases ──────────────────────────────────────────────────────
PHASES = {
    1: {
        'name'        : 'Fondations factuelles',
        'total_tokens': 2_000_000_000,
        'chunks'      : 2,
        'datasets'    : [
            {
                'name'            : 'cosmopedia_v2',
                'source'          : 'HuggingFaceTB/cosmopedia-v2',
                'config'          : 'cosmopedia-v2',
                'split'           : 'train',
                'text_key'        : 'text',
                'tokens_per_chunk': 400_000_000,  # 40%
                'lang_filter'     : 'none',
                'skip_filter'     : True,
            },
            {
                'name'            : 'fineweb_edu',
                'source'          : 'HuggingFaceFW/fineweb-edu',
                'config'          : 'sample-100BT',
                'split'           : 'train',
                'text_key'        : 'text',
                'tokens_per_chunk': 300_000_000,  # 30%
                'lang_filter'     : 'field',
                'skip_filter'     : False,
                'int_score_min'   : 4,
            },
            {
                'name'            : 'pes2o',
                'source'          : 'allenai/olmo-mix-1124',
                'config'          : 'pes2o',
                'split'           : 'train',
                'text_key'        : 'text',
                'tokens_per_chunk': 150_000_000,  # 15%
                'lang_filter'     : 'none',
                'skip_filter'     : True,
            },
            {
                'name'            : 'finemath_4plus',
                'source'          : 'HuggingFaceTB/finemath',
                'config'          : 'finemath-4plus',
                'split'           : 'train',
                'text_key'        : 'text',
                'tokens_per_chunk': 100_000_000,  # 10%
                'lang_filter'     : 'none',
                'skip_filter'     : True,
            },
            {
                'name'            : 'zyda2',
                'source'          : 'Zyphra/Zyda-2',
                'config'          : 'default',
                'split'           : 'train',
                'text_key'        : 'text',
                'tokens_per_chunk': 50_000_000,   # 5%
                'lang_filter'     : 'none',
                'skip_filter'     : False,
            },
        ],
    },

    2: {
        'name'        : 'Factuel + Raisonnement naturel + Code intro',
        'total_tokens': 2_000_000_000,
        'chunks'      : 2,
        'datasets'    : [
            {
                'name'            : 'cosmopedia_v2',
                'source'          : 'HuggingFaceTB/cosmopedia-v2',
                'config'          : 'cosmopedia-v2',
                'split'           : 'train',
                'text_key'        : 'text',
                'tokens_per_chunk': 300_000_000,  # 30%
                'lang_filter'     : 'none',
                'skip_filter'     : True,
            },
            {
                'name'            : 'fineweb_edu',
                'source'          : 'HuggingFaceFW/fineweb-edu',
                'config'          : 'sample-100BT',
                'split'           : 'train',
                'text_key'        : 'text',
                'tokens_per_chunk': 220_000_000,  # 22%
                'lang_filter'     : 'field',
                'skip_filter'     : True,
                'int_score_min'   : 4,
            },
            {
                'name'            : 'pes2o',
                'source'          : 'allenai/olmo-mix-1124',
                'config'          : 'pes2o',
                'split'           : 'train',
                'text_key'        : 'text',
                'tokens_per_chunk': 180_000_000,  # 18%
                'lang_filter'     : 'none',
                'skip_filter'     : True,
            },
            {
                'name'            : 'finemath_4plus',
                'source'          : 'HuggingFaceTB/finemath',
                'config'          : 'finemath-4plus',
                'split'           : 'train',
                'text_key'        : 'text',
                'tokens_per_chunk': 200_000_000,  # 20%
                'lang_filter'     : 'none',
                'skip_filter'     : True,
            },
            {
                'name'            : 'swallowcode_v2',
                'source'          : 'tokyotech-llm/swallow-code-v2',
                'config'          : 'stage5-auto-format',
                'split'           : 'train',
                'text_key'        : 'text',
                'tokens_per_chunk': 100_000_000,  # 10%
                'lang_filter'     : 'none',
                'skip_filter'     : True,
            },
        ],
    },

    3: {
        'name'        : 'Code + Math + Annealing',
        'total_tokens': 1_000_000_000,
        'chunks'      : 1,
        'datasets'    : [
            {
                'name'            : 'finemath_4plus',
                'source'          : 'HuggingFaceTB/finemath',
                'config'          : 'finemath-4plus',
                'split'           : 'train',
                'text_key'        : 'text',
                'tokens_per_chunk': 350_000_000,  # 35%
                'lang_filter'     : 'none',
                'skip_filter'     : True,
            },
            {
                'name'            : 'swallowcode_v2',
                'source'          : 'tokyotech-llm/swallow-code-v2',
                'config'          : 'stage5-auto-format',
                'split'           : 'train',
                'text_key'        : 'text',
                'tokens_per_chunk': 250_000_000,  # 25%
                'lang_filter'     : 'none',
                'skip_filter'     : True,
            },
            {
                'name'            : 'cosmopedia_v2',
                'source'          : 'HuggingFaceTB/cosmopedia-v2',
                'config'          : 'cosmopedia-v2',
                'split'           : 'train',
                'text_key'        : 'text',
                'tokens_per_chunk': 200_000_000,  # 20%
                'lang_filter'     : 'none',
                'skip_filter'     : True,
            },
            {
                'name'            : 'pes2o',
                'source'          : 'allenai/olmo-mix-1124',
                'config'          : 'pes2o',
                'split'           : 'train',
                'text_key'        : 'text',
                'tokens_per_chunk': 200_000_000,  # 20%
                'lang_filter'     : 'none',
                'skip_filter'     : True,
            },
        ],
    },
}

# ── Config ───────────────────────────────────────────────────────────────────
CONFIG = {
    'tokenizer_name'  : 'HuggingFaceTB/cosmo2-tokenizer',
    'output_dir'      : './data',
    'offsets_file'    : './data/gnt_offsets.json',
    'tokens_per_chunk': 1_000_000_000,   # 1B par chunk
    'token_tolerance' : 10_000_000,
    'batch_docs'      : 500,
    'min_text_len'    : 100,
    'flush_every'     : 100_000_000,     # flush disque tous les 100M tokens (~400MB RAM max)
}


def load_offsets(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_offsets(path: str, offsets: dict):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(offsets, f, indent=2)
    os.replace(tmp, path)


def load_tokenizer_with_special():
    from transformers import AutoTokenizer
    print(f'\nTokenizer : {CONFIG["tokenizer_name"]}')
    tok = AutoTokenizer.from_pretrained(CONFIG['tokenizer_name'])

    added = tok.add_special_tokens(
        {'additional_special_tokens': SPECIAL_TOKENS}
    )
    print(f'  vocab={len(tok)}  +{added} tokens spéciaux')
    for t in SPECIAL_TOKENS:
        tid = tok.convert_tokens_to_ids(t)
        ids = tok.encode(t, add_special_tokens=False)
        status = '✅' if len(ids) == 1 else '❌ DÉCOUPÉ'
        print(f'  {status} {t!r:12} → id={tid}')

    tok_dir = Path(CONFIG['output_dir']) / 'tokenizer_gnt'
    tok_dir.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(str(tok_dir))
    print(f'  Tokenizer sauvé → {tok_dir}')
    return tok


def download_dataset_chunk(
    phase_id   : int,
    chunk_id   : int,
    dataset_cfg: dict,
    target_toks: int,
    doc_offset : int,
    tokenizer,
    output_path: Path,
) -> int:
    """
    Télécharge et tokenise un dataset jusqu'à target_toks tokens.
    Flush sur disque tous les flush_every tokens pour éviter d'exploser la RAM.
    RAM max active : ~400MB (flush_every=100M tokens × 4 bytes).
    """
    from datasets import load_dataset
    from tqdm import tqdm

    print(f'\n  Dataset : {dataset_cfg["name"]}')
    print(f'  Target  : {target_toks / 1e6:.0f}M tokens')
    print(f'  Offset  : {doc_offset:,} docs')

    ds = load_dataset(
        dataset_cfg['source'],
        dataset_cfg.get('config'),
        split     = dataset_cfg['split'],
        streaming = True,
    )

    if doc_offset > 0:
        ds = ds.skip(doc_offset)

    # ── Flush progressif sur disque ──────────────────────────────────────────
    # Au lieu d'accumuler all_tokens en RAM (→ OOM sur 400M tokens),
    # on écrit dans un fichier .raw binaire tous les flush_every tokens,
    # puis on convertit en .npy une seule fois à la fin.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_raw  = str(output_path) + '.raw'
    flush_fd = open(tmp_raw, 'wb')

    buf           = []       # buffer RAM courant
    total_flushed = 0        # tokens déjà écrits sur disque
    docs_read     = 0
    batch         = []
    t0            = time.time()

    def flush_buf():
        nonlocal total_flushed
        if not buf:
            return
        arr = np.array(buf, dtype=np.int32)
        arr.tofile(flush_fd)
        total_flushed += len(buf)
        buf.clear()
        gc.collect()

    pbar = tqdm(
        total      = target_toks,
        unit       = 'tok',
        unit_scale = True,
        desc       = f'  {dataset_cfg["name"]}',
        dynamic_ncols=True,
        colour     = 'cyan',
    )

    for doc in ds:
        text = doc.get(dataset_cfg['text_key'], '') or ''
        if len(text) < CONFIG['min_text_len']:
            continue

        mode = dataset_cfg.get('lang_filter', 'none')
        if mode == 'field' and doc.get('language', 'en') != 'en':
            continue
        if mode == 'fasttext':
            score = doc.get('language_id_whole_page_fasttext', 1.0)
            if score < 0.65:
                continue

        score_min = dataset_cfg.get('int_score_min', 0)
        if score_min > 0 and doc.get('int_score', score_min) < score_min:
            continue

        batch.append(text)
        docs_read += 1

        if len(batch) >= CONFIG['batch_docs']:
            prev = total_flushed + len(buf)
            for t in batch:
                ids = tokenizer.encode(t, add_special_tokens=False)
                ids.append(tokenizer.eos_token_id)
                buf.extend(ids)
            pbar.update((total_flushed + len(buf)) - prev)
            batch = []

            # Flush si le buffer RAM est assez grand
            if len(buf) >= CONFIG['flush_every']:
                flush_buf()

            if total_flushed + len(buf) >= target_toks + CONFIG['token_tolerance']:
                break

    # Flush le batch + buf restants
    for t in batch:
        ids = tokenizer.encode(t, add_special_tokens=False)
        ids.append(tokenizer.eos_token_id)
        buf.extend(ids)
    flush_buf()

    flush_fd.close()
    pbar.update(total_flushed - pbar.n)
    pbar.close()

    elapsed = time.time() - t0
    print(f'  Collecté : {total_flushed/1e6:.0f}M tokens  ({elapsed/60:.1f}min)')
    print(f'  Docs lus : {docs_read:,}')

    # ── Conversion .raw → .npy, tronqué au target exact ─────────────────────
    print(f'  Conversion .raw → .npy...')
    arr = np.fromfile(tmp_raw, dtype=np.int32)
    arr = arr[:target_toks]
    np.save(str(output_path).replace('.npy', ''), arr)
    os.remove(tmp_raw)
    del arr
    gc.collect()
    print(f'  Sauvé   : {output_path}  ({output_path.stat().st_size/1e9:.2f}GB)')

    return doc_offset + docs_read


def shuffle_chunk(combined: np.ndarray, seq_len: int = 1024) -> np.ndarray:
    """
    Shuffle au niveau des séquences (pas des tokens individuels).

    On découpe `combined` en blocs de `seq_len` tokens, on mélange
    ces blocs, puis on reconstruit le tableau. Les tokens restants
    (dernier bloc incomplet) sont conservés à la fin sans mélange.
    Le shuffle est non-déterministe : le fichier .npy est écrit une
    seule fois, donc la reproductibilité d'un seed fixe est inutile.
    """
    n_complete = len(combined) // seq_len
    remainder  = len(combined) %  seq_len

    indices = np.arange(n_complete)
    rng = np.random.default_rng()   # seed aléatoire — inutile d'être déterministe
    rng.shuffle(indices)

    shuffled_blocks = combined[: n_complete * seq_len].reshape(n_complete, seq_len)
    shuffled_blocks = shuffled_blocks[indices]
    result = shuffled_blocks.reshape(-1)

    if remainder > 0:
        tail = combined[n_complete * seq_len :]
        result = np.concatenate([result, tail])

    return result


def assemble_chunk(
    chunk_dir       : Path,
    chunk_file      : Path,
    phase_cfg_dsets : list,
    global_chunk_id : int,
):
    """Concatène les fichiers par dataset, shuffle au niveau séquence, sauvegarde."""
    all_chunk_tokens = []
    for ds_cfg in phase_cfg_dsets:
        ds_file = chunk_dir / f'{ds_cfg["name"]}.npy'
        if ds_file.exists():
            arr = np.load(str(ds_file), mmap_mode='r')
            all_chunk_tokens.append(arr)

    if not all_chunk_tokens:
        return

    combined = np.concatenate(all_chunk_tokens)[:CONFIG['tokens_per_chunk']]
    combined = shuffle_chunk(combined, seq_len=1024)

    np.save(str(chunk_file).replace('.npy', ''), combined)
    print(f'  ✅ chunk_{global_chunk_id:03d} → {len(combined)/1e9:.3f}B tokens (shuffled)')

    for ds_cfg in phase_cfg_dsets:
        ds_file = chunk_dir / f'{ds_cfg["name"]}.npy'
        if ds_file.exists():
            ds_file.unlink()


def download_dry_chunk(tokenizer):
    """
    Génère le dry_chunk : 1B tokens issus de Cosmopedia v2 uniquement.
    Utile pour valider le pipeline (tokenizer, offsets, shuffle) avant
    de lancer les phases complètes.
    """
    print(f'\n{"="*60}')
    print(f'  DRY CHUNK — {DRY_CHUNK["label"]}')
    print(f'{"="*60}')

    offsets   = load_offsets(CONFIG['offsets_file'])
    chunk_dir  = Path(CONFIG['output_dir']) / 'dry_chunk'
    chunk_file = chunk_dir / 'tokens.npy'

    if chunk_file.exists():
        arr = np.load(str(chunk_file), mmap_mode='r')
        if abs(len(arr) - CONFIG['tokens_per_chunk']) <= CONFIG['token_tolerance']:
            print(f'  ✅ dry_chunk déjà OK ({len(arr)/1e9:.3f}B tokens) — skip')
            return

    ds_cfg    = DRY_CHUNK['datasets'][0]
    ds_key    = f'dry_{ds_cfg["name"]}_offset'
    ds_offset = offsets.get(ds_key, 0)
    ds_file   = chunk_dir / f'{ds_cfg["name"]}.npy'

    if not ds_file.exists():
        new_offset = download_dataset_chunk(
            phase_id    = 0,
            chunk_id    = 0,
            dataset_cfg = ds_cfg,
            target_toks = ds_cfg['tokens_per_chunk'],
            doc_offset  = ds_offset,
            tokenizer   = tokenizer,
            output_path = ds_file,
        )
        offsets[ds_key] = new_offset
        save_offsets(CONFIG['offsets_file'], offsets)

    assemble_chunk(
        chunk_dir       = chunk_dir,
        chunk_file      = chunk_file,
        phase_cfg_dsets = DRY_CHUNK['datasets'],
        global_chunk_id = -1,
    )

    offsets['dry_chunk_done'] = True
    save_offsets(CONFIG['offsets_file'], offsets)
    print(f'  dry_chunk → {chunk_file}')


def download_phase(phase_id: int, phase_cfg: dict, tokenizer):
    """Télécharge un chunk de 1B tokens pour la phase donnée."""
    print(f'\n{"="*60}')
    print(f'  PHASE {phase_id} — {phase_cfg["name"]}')
    print(f'{"="*60}')

    offsets = load_offsets(CONFIG['offsets_file'])
    phase_key = f'phase_{phase_id}'

    done_chunks   = offsets.get(f'{phase_key}_chunks_done', 0)
    target_chunks = phase_cfg['chunks']

    if done_chunks >= target_chunks:
        print(f'  ✅ Phase {phase_id} déjà complète ({done_chunks}/{target_chunks} chunks)')
        return

    for chunk_id in range(done_chunks, target_chunks):
        global_chunk_id = sum(
            PHASES[p]['chunks'] for p in range(1, phase_id)
        ) + chunk_id

        chunk_dir  = Path(CONFIG['output_dir']) / f'chunk_{global_chunk_id:03d}'
        chunk_file = chunk_dir / 'tokens.npy'

        if chunk_file.exists():
            arr = np.load(str(chunk_file), mmap_mode='r')
            if abs(len(arr) - CONFIG['tokens_per_chunk']) <= CONFIG['token_tolerance']:
                print(f'  ✅ chunk_{global_chunk_id:03d} OK ({len(arr)/1e9:.3f}B) — skip')
                continue

        print(f'\n  Chunk {global_chunk_id} (Phase {phase_id}, chunk {chunk_id+1}/{target_chunks})')

        for ds_cfg in phase_cfg['datasets']:
            ds_key     = f'{phase_key}_{ds_cfg["name"]}_offset'
            ds_offset  = offsets.get(ds_key, 0)
            target_ds  = ds_cfg['tokens_per_chunk']

            ds_file = chunk_dir / f'{ds_cfg["name"]}.npy'

            if not ds_file.exists():
                new_offset = download_dataset_chunk(
                    phase_id    = phase_id,
                    chunk_id    = chunk_id,
                    dataset_cfg = ds_cfg,
                    target_toks = target_ds,
                    doc_offset  = ds_offset,
                    tokenizer   = tokenizer,
                    output_path = ds_file,
                )
                offsets[ds_key] = new_offset
                save_offsets(CONFIG['offsets_file'], offsets)

        assemble_chunk(
            chunk_dir       = chunk_dir,
            chunk_file      = chunk_file,
            phase_cfg_dsets = phase_cfg['datasets'],
            global_chunk_id = global_chunk_id,
        )

        offsets[f'{phase_key}_chunks_done'] = chunk_id + 1
        save_offsets(CONFIG['offsets_file'], offsets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, choices=[1, 2, 3],
                        help='Télécharge uniquement cette phase (défaut: toutes)')
    args = parser.parse_args()

    print('=' * 60)
    print('  GNT — Dataset Downloader')
    print('  Curriculum : 5B tokens / 3 phases + dry_chunk 1B')
    print(f'  Tokens spéciaux : {SPECIAL_TOKENS}')
    print('=' * 60)

    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    tokenizer = load_tokenizer_with_special()

    download_dry_chunk(tokenizer)

    phases_to_run = [args.phase] if args.phase else [1, 2, 3]

    for phase_id in phases_to_run:
        download_phase(phase_id, PHASES[phase_id], tokenizer)

    print('\n✅ Download terminé')
    print(f'Tokenizer étendu : {CONFIG["output_dir"]}/tokenizer_gnt/')
    print(f'Offsets          : {CONFIG["offsets_file"]}')


if __name__ == '__main__':
    main()