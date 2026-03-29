#!/usr/bin/env python3
"""
dataset_exp.py — GNT Experiment
=================================
30 chunks x 1B tokens — Cosmopedia-v2 uniquement.
Tokenizer : HuggingFaceTB/cosmo2-tokenizer (base, sans tokens spéciaux)
Pas de shuffle (dataset homogène, inutile).

USAGE :
  python3.10 dataset_exp.py
"""

import os
import sys
import json
import time
import argparse
import gc
import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    'tokenizer_name'  : 'HuggingFaceTB/cosmo2-tokenizer',
    'output_dir'      : './data_exp',
    'offsets_file'    : './data_exp/exp_offsets.json',
    'tokens_per_chunk': 1_000_000_000,   # 1B par chunk
    'token_tolerance' : 10_000_000,
    'n_chunks'        : 3,
    'batch_docs'      : 500,
    'min_text_len'    : 100,
    'flush_every'     : 100_000_000,  # flush disque tous les 100M tokens
    'dataset': {
        'source'  : 'HuggingFaceTB/cosmopedia-v2',
        'config'  : 'cosmopedia-v2',
        'split'   : 'train',
        'text_key': 'text',
    },
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


def load_tokenizer():
    from transformers import AutoTokenizer
    print(f'\nTokenizer : {CONFIG["tokenizer_name"]}')
    tok = AutoTokenizer.from_pretrained(CONFIG['tokenizer_name'])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f'  vocab={len(tok)}  eos={tok.eos_token_id}')
    return tok


def download_chunk(chunk_id: int, doc_offset: int, tokenizer) -> int:
    from datasets import load_dataset
    from tqdm import tqdm

    ds_cfg     = CONFIG['dataset']
    target     = CONFIG['tokens_per_chunk']
    chunk_dir  = Path(CONFIG['output_dir']) / f'chunk_{chunk_id:03d}'
    chunk_file = chunk_dir / 'tokens.npy'
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # Resume check
    if chunk_file.exists():
        arr = np.load(str(chunk_file), mmap_mode='r')
        if abs(len(arr) - target) <= CONFIG['token_tolerance']:
            print(f'  ✅ chunk_{chunk_id:03d} déjà OK ({len(arr)/1e9:.3f}B) — skip')
            return doc_offset

    print(f'\n{"="*60}')
    print(f'  Chunk {chunk_id:03d} / {CONFIG["n_chunks"]-1}')
    print(f'  Target  : {target/1e9:.1f}B tokens')
    print(f'  Offset  : {doc_offset:,} docs')
    print(f'{"="*60}')

    ds = load_dataset(
        ds_cfg['source'],
        ds_cfg['config'],
        split     = ds_cfg['split'],
        streaming = True,
    )
    if doc_offset > 0:
        print(f'  Skip {doc_offset:,} docs...')
        ds = ds.skip(doc_offset)

    # Flush progressif sur disque — RAM buf max ~400MB
    tmp_raw  = str(chunk_file) + '.raw'
    flush_fd = open(tmp_raw, 'wb')

    buf           = []
    total_flushed = 0
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
        total        = target,
        unit         = 'tok',
        unit_scale   = True,
        desc         = f'  chunk_{chunk_id:03d}',
        dynamic_ncols= True,
        colour       = 'cyan',
    )

    for doc in ds:
        text = doc.get(ds_cfg['text_key'], '') or ''
        if len(text) < CONFIG['min_text_len']:
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

            if len(buf) >= CONFIG['flush_every']:
                flush_buf()

            if total_flushed + len(buf) >= target + CONFIG['token_tolerance']:
                break

    # flush batch + buf restants
    for t in batch:
        ids = tokenizer.encode(t, add_special_tokens=False)
        ids.append(tokenizer.eos_token_id)
        buf.extend(ids)
    flush_buf()

    flush_fd.close()
    pbar.update(total_flushed - pbar.n)
    pbar.close()

    elapsed = time.time() - t0
    print(f'  Collecté : {total_flushed/1e9:.3f}B tokens  ({elapsed/60:.1f}min)')
    print(f'  Docs lus : {docs_read:,}')

    # Conversion .raw -> .npy, tronqué au target exact
    print(f'  Conversion .raw -> .npy...')
    arr = np.fromfile(tmp_raw, dtype=np.int32)
    arr = arr[:target]
    np.save(str(chunk_file).replace('.npy', ''), arr)
    os.remove(tmp_raw)
    del arr; gc.collect()
    print(f'  Sauvé    : {chunk_file}  ({chunk_file.stat().st_size/1e9:.2f}GB)')

    return doc_offset + docs_read


def main():
    print('=' * 60)
    print('  GNT Experiment — Dataset Downloader')
    print('  30B tokens — Cosmopedia-v2 uniquement')
    print('  Pas de tokens spéciaux, pas de shuffle')
    print('=' * 60)

    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    tokenizer = load_tokenizer()
    offsets   = load_offsets(CONFIG['offsets_file'])

    doc_offset = offsets.get('doc_offset', 0)

    for chunk_id in range(CONFIG['n_chunks']):
        chunk_file = Path(CONFIG['output_dir']) / f'chunk_{chunk_id:03d}' / 'tokens.npy'

        # Si le chunk existe déjà et est valide, on passe sans toucher à l'offset
        if chunk_file.exists():
            arr = np.load(str(chunk_file), mmap_mode='r')
            if abs(len(arr) - CONFIG['tokens_per_chunk']) <= CONFIG['token_tolerance']:
                print(f'  ✅ chunk_{chunk_id:03d} déjà OK — skip')
                continue

        doc_offset = download_chunk(chunk_id, doc_offset, tokenizer)
        offsets['doc_offset']    = doc_offset
        offsets['chunks_done']   = chunk_id + 1
        save_offsets(CONFIG['offsets_file'], offsets)

    print('\n✅ Download terminé — 30 chunks x 1B tokens')
    print(f'  Data : {CONFIG["output_dir"]}/')
    print(f'  Offsets : {CONFIG["offsets_file"]}')


if __name__ == '__main__':
    main()