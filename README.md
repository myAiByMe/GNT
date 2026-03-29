# GNT v1 — Graph Neural Transformer

> *Mémoire récurrente + attention relationnelle : quand le Transformer apprend à naviguer un graphe.*

---

## Qu'est-ce que GNT ?

Les Transformers classiques ont un défaut structurel : chaque token est un vecteur **isolé**. La mémoire factuelle émerge uniquement par répétition statistique dans les données — ce qui explique pourquoi un modèle de 7B paramètres "sait" plus de faits qu'un modèle de 700M, non pas parce qu'il est plus intelligent, mais parce qu'il a plus de neurones pour mémoriser.

**GNT attaque ce problème sur deux fronts simultanément :**

1. **GatedDeltaNet (GDN)** — une mémoire récurrente par couche qui maintient un état matriciel `S_t` mis à jour à chaque token via une règle d'apprentissage en ligne (Gated Delta Rule). Le modèle peut stocker et récupérer des associations clé→valeur dynamiquement, sans dépendre de la répétition dans les données.

2. **NaylisAttention** — une attention hybride qui injecte un biais de graphe asymétrique `B[i,j] = <R_q(i), R_k(j)>` dans les scores d'attention, permettant de coder des directions sémantiques (Paris→France ≠ France→Paris).

---

## Architecture

### Vue d'ensemble

GNT est une architecture hybride **75% GDN + 25% Naylis** : sur 18 layers, 14 sont des GDNBlocks et 4 sont des NaylisAttnBlocks (un tous les 4 layers).

```
Input [B, S]
    ↓
token_embeddings [B, S, 768]
    ↓
┌─────────────────────────────────────────┐
│  Layer 0   →  GDNBlock                  │
│  Layer 1   →  GDNBlock                  │
│  Layer 2   →  GDNBlock                  │
│  Layer 3   →  NaylisAttnBlock  ◄──────  │ (attn_every_n_layers=4)
│  Layer 4   →  GDNBlock                  │
│  ...                                    │
│  Layer 17  →  NaylisAttnBlock           │
└─────────────────────────────────────────┘
    ↓
RMSNorm → output_head [B, S, vocab_size]
```

---

### GDNBlock — Mémoire récurrente (75% des layers)

Le GDNBlock implémente la **Gated Delta Rule** : une mémoire matricielle `S_t ∈ R^{d×d}` par head, mise à jour token par token.

```
Règle : S_t = α_t · S_{t-1} + β_t · (v_t - S_{t-1} k_t) ⊗ k_t
```

- `α_t` (forget gate) : contrôle combien de la mémoire précédente est conservée. Init `sigmoid(+2.0) ≈ 0.88` — mémoire stable au départ.
- `β_t` (learn gate) : contrôle la force de la mise à jour. Init `sigmoid(0.0) = 0.5`.
- La sortie `o_t = S_t · q_t` récupère ce que la mémoire prédit pour la query courante.

Le biais Naylis est également injecté dans le GDN via les vecteurs `R_q` / `R_k` additionnés aux queries/keys avant normalisation L2.

**Backend :**
- FLA `chunk_gated_delta_rule` (Triton) si disponible — **249× plus rapide**, **6× moins de VRAM** que le fallback Python.
- Fallback Python correct si FLA absent (~30h/5B tokens).

---

### NaylisAttnBlock — Attention relationnelle (25% des layers)

```
x [B, S, 768]
├── q_proj / k_proj / v_proj  →  attention classique (SDPA)
│       + QK-Norm + Partial RoPE (25% des dims)
└── rel_q_proj / rel_k_proj   →  R_q [B, H, S, R]  R_k [B, H, S, R]
                                        ↓
                               B[i,j] = <R_q(i), R_k(j)>   ← ASYMÉTRIQUE
                                        ↓
                               scores = QKᵀ/√d + graph_scale · B
```

**Asymétrie directionnelle :** deux projecteurs séparés garantissent `B[i,j] ≠ B[j,i]`.  
La relation "Paris → capitale → France" est distincte de "France → capitale → Paris".

**graph_scale — Activation progressive :**
```python
self.graph_scale = nn.Parameter(torch.zeros(num_heads))
```
Initialisé à **zéro par head** : au step 0, le modèle est un Transformer classique stable.  
Chaque head décide indépendamment d'activer ses canaux relationnels selon le signal de gradient.

**Partial RoPE (25% des dims) :** le GDN encode déjà la position via sa récurrence. Appliquer RoPE sur 100% des dims créerait un conflit de signal — 25% suffit pour l'encodage positionnel sans écraser les 75% de dims sémantiques.

---

## Stack Technique

| Composant | Choix | Raison |
|---|---|---|
| Attention | SDPA (PyTorch 2.x) prioritaire | FA natif Blackwell SM120 via PyTorch 2.8 |
| Attention varlen | flash_attn_varlen_func | Sequence packing — 0% padding |
| GDN kernel | FLA chunk_gated_delta_rule | 249× plus rapide que Python, correctness validée |
| Embeddings | Partial RoPE (25%) + QK-Norm | Stabilité + pas de conflit avec récurrence GDN |
| FFN | SwiGLU | +15% vs GELU sur benchmarks |
| GQA | n_kv_heads=4 (ratio 3:1) | −25% mémoire KV cache |
| Tokenizer | cosmo2-tokenizer (49 152 tokens) | Cohérence données/vocab |
| Optimiseur | Muon+MARS (blocs) + AdamW (embeddings) | Convergence ~2× plus rapide que AdamW seul |
| Scheduler | WSD (Warmup-Stable-Decay) | Standard moderne SLM |
| Data pretrain | Cosmopedia v2 — 5B tokens (5 chunks × 1B) | Textbook synthétique dense en faits structurés |
| RAM data | mmap numpy chunks | ~400MB RAM actif par chunk |
| Gradient | Gradient checkpointing (optionnel) | −60% VRAM, +30% temps |

---

## Configuration GNT v1 (~200M params)

```python
vocab_size          = 49_152   # cosmo2-tokenizer
embed_dim           = 768
num_heads           = 12
n_kv_heads          = 4        # GQA
num_layers          = 18
rel_rank            = 32       # dims canaux relationnels par head
max_seq_len         = 1024
attn_every_n_layers = 4        # 1 NaylisAttn tous les 4 layers
conv_kernel         = 4        # ShortConv1d dans GDN
dropout             = 0.0
```

**Décomposition paramètres :**
```
token_embeddings      : 49 152 × 768          =  37.7M
14 × GDNBlock         : ~8.5M  × 14           = 119.0M
 4 × NaylisAttnBlock  : ~9.5M  × 4            =  38.0M
  dont Naylis         : rel_q/k_proj + scale  =  ~2.4% du total
ln_final              : 768                   =   0.6K
──────────────────────────────────────────────────────
Total                                          ≈ 195M
```

---

## Hyperparamètres d'entraînement

### Pretrain (5B tokens)

```python
batch_size          = 2
gradient_accumulation = 8      # batch effectif = 16
learning_rate       = 4e-4
weight_decay        = 0.1
adam_beta1          = 0.9
adam_beta2          = 0.95
max_grad_norm       = 1.0
warmup_ratio        = 0.03
decay_ratio         = 0.15
min_lr_ratio        = 0.1      # lr finale = 4e-5
val_split_tokens    = 10_000_000  # 10M tokens par chunk réservés à la val
```

### SFT Phase A — Think + Conversationnel

```python
lr          = 2e-5
epochs      = 1
batch_size  = 2
grad_accum  = 16    # batch effectif = 32
max_samples = 63_000
```

Datasets : smol-smoltalk (30%) · DailyDialog (10%) · Mixture-of-Thoughts (50%) · NuminaMath (10%) · MMLU (5%)

Loss masking : loss uniquement sur les tokens du bloc `assistant` (think inclus, poids uniformes).

### SFT Phase B — Sandbox Python

```python
lr          = 5e-6
epochs      = 1
batch_size  = 2
grad_accum  = 16
max_samples = 40_000
```

Datasets : Magicoder-Evol-Instruct (40%) · CodeFeedback (35%) · Python-Edu-Distilled (25%)

Loss pondérée : `<code>` × 1.5 · `<o>` × 0.5 · reste × 1.0

---

## Tokens Spéciaux

```
<|im_start|>  <|im_end|>    # ChatML — structure des turns
<think>  </think>            # bloc de raisonnement (chain-of-thought)
<code>   </code>             # code Python (sandbox)
<o>      </o>               # output d'exécution Python
```

**Format ChatML GNT :**
```
<|im_start|>system
You are GNT...<|im_end|>
<|im_start|>user
Question<|im_end|>
<|im_start|>assistant
<think>raisonnement...</think>
Réponse<|im_end|>
```

---

## Pourquoi Cosmopedia v2 ?

Cosmopedia v2 est du contenu textbook synthétique dense en faits structurés. Chaque document reformule les mêmes relations factuelles de nombreuses façons :

```
"Paris is the capital of France."
"France's capital city, Paris, ..."
"Located in northern France, Paris serves as the country's capital..."
```

Ce signal répété sous angles multiples est exactement ce dont les canaux relationnels GDN+Naylis ont besoin — ils verront la relation Paris→France sous assez de formes pour que `B[Paris, France] > B[Paris, Germany]` et que la mémoire GDN associe `France → capital → Paris` de façon stable.

---

## Usage

```bash
# 1. Télécharger et tokeniser les données (5 chunks × 1B tokens)
python3.10 dataset_gnt.py --phase 1

# 2. Lancer le pretrain
python3.10 pretrain_gnt.py

# 3. Sans torch.compile (si problème de compilation)
python3.10 pretrain_gnt.py --no-compile

# 4. SFT (Phase A puis B automatiquement)
python3.10 sft_gnt.py

# 5. SFT Phase A seulement
python3.10 sft_gnt.py --only-phase A

# 6. Benchmarks autonomes
python3.10 benchmark_gnt.py --ckpt ./Model/gnt_pretrain.pt
python3.10 benchmark_gnt.py --ckpt ./Model/gnt_sft_phaseB.pt --mode sft --label B
```

---

## Benchmarks

### Pretrain (log-likelihood, comparable aux papiers)
HellaSwag · ARC-Easy · ARC-Challenge (25-shot) · WinoGrande · PIQA · MMLU (5-shot) · TriviaQA

### SFT (pretrain + instruct)
IFEval (instruction following strict) · GSM8K (raisonnement mathématique CoT, 5-shot) · TruthfulQA MC1

Les benchmarks sont lancés automatiquement après chaque phase SFT et sauvegardés dans `./Model/`.

---

## Structure du projet

```
GNT/
├── Core/
│   ├── Attention/
│   │   ├── attention.py          # NaylisAttention (Transformer classique + graph bias)
│   │   └── gdn_attention.py      # GDNNaylisAttention (GatedDeltaNet + graph bias)
│   ├── FeedForward/
│   │   └── feedforward.py        # SwiGLU FFN
│   ├── Model/
│   │   └── GNT.py                # Modèle principal — hybride GDN+Naylis
│   └── TransformerBlock/
│       ├── attn_block.py         # NaylisAttnBlock (PreNorm + Naylis + FFN)
│       └── gdn_block.py          # GDNBlock (PreNorm + GDN + FFN)
├── data/
│   ├── chunk_000/tokens.npy      # ~1B tokens (mmap)
│   ├── chunk_001/tokens.npy
│   ├── chunk_002/tokens.npy
│   ├── chunk_003/tokens.npy
│   ├── chunk_004/tokens.npy
│   └── tokenizer_gnt/            # cosmo2-tokenizer + tokens spéciaux GNT
├── Model/                        # checkpoints + graphs de loss + benchmarks
│   ├── gnt_pretrain.pt
│   ├── gnt_sft_phaseA.pt
│   ├── gnt_sft_phaseB.pt
│   ├── gnt_loss_window.png
│   ├── gnt_loss_ema.png
│   └── gnt_benchmarks_*.png
├── CompileCache/                 # cache torch.compile (FX graph)
├── pretrain_gnt.py               # entraînement pretrain — 5B tokens
├── sft_gnt.py                    # SFT Phase A (think) + Phase B (code)
├── benchmark_gnt.py              # évaluation lm-eval-harness
└── README.md
```

---

## Analyse de risques

### Ce qui est solide ✅

**graph_scale init=0** est la protection principale. Le modèle commence comme un Transformer classique stable — si les canaux relationnels n'apportent rien, le gradient ne les activera pas.

**L'asymétrie** (`rel_q_proj` ≠ `rel_k_proj`) est mathématiquement correcte pour coder des directions sémantiques.

**La Gated Delta Rule** est validée dans la littérature (DeltaNet, GLA) avec un alpha correctement appliqué sur `S` — le bug historique (alpha encodé dans `v`) est corrigé.

**FLA validé empiriquement** : erreur relative Python vs FLA kernel = 0.0237 sur les activations — correctness confirmée.

### Ce qui est incertain ⚠️

**Le signal va-t-il survivre ?** La question centrale reste : est-ce que le gradient va activer `graph_scale` ? Surveiller `|graph_scale|` moyen pendant l'entraînement — si < 1e-4 après 1B tokens, le signal est trop faible.

**Interaction GDN × Naylis** : les deux mécanismes injectent le biais relationnel (`R_q`, `R_k`) différemment. En GDN il est fusionné dans `k_tilde` avant la règle delta ; en Naylis il est ajouté aux scores d'attention. L'effet combiné sur 18 layers n'est pas prévisible analytiquement.

**Généralisation** : même si les canaux émergent sur Cosmopedia, ils devront se généraliser en SFT sur des distributions plus variées.

### Estimation réaliste

| Scénario | Probabilité |
|---|---|
| GDN + Naylis émergent, gain factuel +15-30% vs baseline | ~50% |
| Émergence partielle, gain < 10% | ~30% |
| Canaux restent silencieux (gradient ne les active pas) | ~15% |
| Instabilité training | ~5% |

---

## Métriques à surveiller pendant le pretrain

```
graph_scale moyen      → doit croître au-delà de 1e-3 après 500M tokens
val loss               → doit descendre sous 3.0 après 2B tokens
train/val gap          → si > 0.3, risque d'overfitting sur les chunks
FLA backend            → vérifier que "FLA chunk_gated_delta_rule : OK" s'affiche
```

---

## Next Steps après pretrain

1. **Mesurer graph_scale par layer** : quels layers ont activé les canaux ? Pattern syntaxique/sémantique/factuel ?
2. **Benchmark factuel** : TriviaQA, NaturalQuestions vs baseline identique sans canaux Naylis (graph_scale gelé à 0)
3. **Visualiser B[i,j]** sur des exemples connus : `B[Paris, France]` vs `B[Paris, Germany]` ?
4. **SFT Phase A** : amplifier le signal relationnel avec des paires Q&A factuelles
5. **SFT Phase B** : sandbox Python — tester si la mémoire GDN aide à suivre l'état d'exécution
6. **Scaler à 500M-1B** : si le gain est réel à 200M, il devrait s'amplifier

---

*GNT v1 — "la mémoire est dans la récurrence, le graphe est dans l'attention."*
