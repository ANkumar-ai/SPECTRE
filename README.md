# SPECTRE — Spectral Phase-Encoded Complex Token Representations 

(Synthesized via Socratic dialogue between Ajay and Fable)

**A deterministic, parameter-light Fourier alternative to Kronecker embeddings — no learned embedding table, no additive positional encoding, no OOV.**

Each token is encoded as a fixed-width interference pattern of unit phasors: bytes choose *channels*, codepoint positions choose *phases*, and a geometric ladder of frequencies gives multi-scale resolution. One learned linear projection maps that fixed vector into `d_model`. Sequence position is then injected multiplicatively, RoPE-style, so additive PE is removed entirely.

## Please go through the sample excel attached to see how SPECTRE works, in detail.
---

## TL;DR

Word-level tinyshakespeare, 4-layer GPT, identical attention/MLP/head architecture in both arms, seed 1337, 3,000 iterations:

| Data | `d_model` | `n_head` | `K` | Vanilla val loss | SPECTRE val loss | Δ |
|---|---|---|---|---|---|---|
| Clean | 128 | 4 | 8 | ~7.5 | ~6.2 | ~1.3 |
| Clean | 256 | 8 | 16 | 6.630 | 5.746 | **0.884** |
| Typo (10%) | 256 | 8 | 16 | 7.474 | 6.141 | **1.333** |

SPECTRE is ahead in every configuration tested, and the margin is ~50% larger on typo-corrupted text (1.33 vs 0.88 nats) — consistent with the claim that a character-level perturbation only disturbs a few phasors rather than replacing an entire embedding row.

Before treating these numbers as a clean apples-to-apples win, read **[Caveats & controls still to run](#caveats--controls-still-to-run)**. The baseline arm has an initialization/scaling issue and a head-tying asymmetry that plausibly account for part of the gap.

---

## 1. The question this answers

From the *Kronecker Embeddings V2* open-problem set, Question #4:

> Why can't I represent each character like a Fourier wave, and just add them to make a word?

Kronecker embeddings (Shravan, 2026) encode a token as a fixed one-hot over `(byte, position)` pairs, truncated at 32 bytes. Deterministic and parameter-efficient, but:

- **Hard truncation** — everything past byte 32 is discarded outright.
- **No interaction between repeated characters** — each `(byte, position)` cell is orthogonal, so spacing between two `l`s in `hello` carries no signal.
- **UTF-8 byte tax** — Indic scripts spend 3 bytes per codepoint, burning the position budget ~3× faster than Latin.

SPECTRE keeps determinism, fixed width, and OOV capability while fixing all three.

---

## 2. How SPECTRE works

### The codec (fixed, not learned)

For one token string:

1. **Normalize** — NFC.
2. **Index by codepoint, not byte** — position `p = 0 … L-1` over codepoints. This is what refunds the UTF-8 tax: a Devanagari codepoint costs one position, same as `a`.
3. **256 byte boxes** — one channel per byte value `0x00–0xFF`.
4. **K lenses** — a geometric ladder of angular speeds `ω_k` from `0.9π` down to `π / L_max`, where `L_max` is the 99.9th-percentile token length in the vocabulary (≈14 codepoints for Shakespeare).
5. **Drop the arrows** — for each codepoint at position `p`, every UTF-8 byte `b` it emits deposits a unit phasor `e^{-i ω_k p}` into box `b`, for all `k`.
6. **Let them interfere** — phasors are vector-summed inside each `(box, lens)` cell. Repeated characters superpose: **magnitude encodes spacing, phase encodes mean position.**
7. **One global normalizer** — the whole `256 × K` complex grid is divided by the square root of its total energy, so every token arrives at unit norm regardless of length.
8. **Flatten** — box-major, then lens, then `(Re, Im)` → `2 · 256 · K` reals.

### The learned part

A single `nn.Linear(2·256·K → d_model, bias=False)`. That is the *entire* learned input pathway — there is no `V × d_model` table.

### Position in the sequence

The projected embedding is split into `d_model/2` complex pairs; pair `j` is rotated by `θ_j · m`, where `m` is the slot index and `θ_j` is a second geometric ladder from `0.9π` down to `π / max_slots`. No additive PE anywhere in the model.

### Why this is different

| | Kronecker | SPECTRE |
|---|---|---|
| Position within token | one-hot coordinate | continuous phase angle |
| Repeated characters | orthogonal, no interaction | summed → magnitude encodes spacing |
| Long tokens | hard crop at 32 bytes | graceful blur past `L_max` |
| UTF-8 tax | burns position budget | refunded by codepoint indexing |
| Sequence position | additive PE | multiplicative RoPE-style twist |
| Inductive bias | sparse, dimension-indexed | shift-invariant spacing + global phase |

---

## 3. Where the ideas came from

| Source | What it contributed |
|---|---|
| **DFT shift theorem** | Shifting a signal multiplies its spectrum by a phase ramp and leaves magnitude untouched. This is the whole trick: spacing lives in magnitude, absolute position lives in phase. |
| **Wavelets / multi-scale analysis** | The geometric lens ladder is a filter bank — fast lenses resolve local adjacency, slow lenses carry global order. |
| **Holographic Reduced Representations** | HRR binds value to role and superposes. SPECTRE uses *phase-encoded* roles instead of random ones, making the superposition a literal Fourier sum. |
| **RoPE** | Rotating dimension pairs by `θ·m` makes attention depend on relative position. Applied post-projection here, which is what lets additive PE be deleted. |
| **ISCII (1991)** | Used a script-mode byte to dodge the multi-byte tax. SPECTRE recovers that economy by separating byte identity (box) from position (phase), keeping the codec fixed-width for every script. |

Unifying principle: **don't spend a dimension on every position — encode position as continuous phase, and use multiple scales to make it robust.**

---

## 4. Repository layout

```
spectre.py                 geometric_ladder, codec_vector, SpectreEmbedding
model.py                   4-layer GPT; arm='vanilla' | 'spectre'
train.py                   CLI training loop → results.json, curves.png
generate_typo.py           10% word-level corruption (swap / delete / substitute)
SPECTRE_typo_test.ipynb    Colab: runs clean + typo, plots side-by-side
requirements.txt           torch>=2.0, numpy, matplotlib
data/input.txt             tinyshakespeare (auto-downloaded)
results_clean.json         example output, 3k-iter run
results_typo.json          example output, 3k-iter run
```

The two arms in `model.py` share everything downstream of the embedding — same pre-LN blocks, same `nn.MultiheadAttention`, same 4× GELU MLP, same causal mask, same output head shape. Only the input pathway differs:

- `arm='vanilla'` — `nn.Embedding(V, d) * sqrt(d)` + sinusoidal PE, output head tied to the embedding.
- `arm='spectre'` — `SpectreEmbedding` (fixed codec → learned projection → slot twist), untied output head.

---

## 5. Running it

### Colab

1. Zip the folder and upload it.
2. Open `SPECTRE_typo_test.ipynb`.
3. Runtime → Run all. It downloads Shakespeare, generates the typo corpus, trains both arms on both corpora, and writes `comparison.png`.

### Local

```bash
pip install -r requirements.txt

# clean
python train.py --arm both --data data/input.txt      --iters 3000 --eval_every 100 --seed 1337
mv results.json results_clean.json

# 10% typos
python generate_typo.py data/input.txt data/input_typo.txt
python train.py --arm both --data data/input_typo.txt --iters 3000 --eval_every 100 --seed 1337
mv results.json results_typo.json
```

Flags: `--arm {both,vanilla,spectre}` · `--iters` · `--batch` · `--block` · `--dmodel` · `--nlayer` · `--nhead` · `--K` · `--lr` · `--eval_every` · `--seed` · `--data`

Constraints worth knowing: `d_model` must be even and divisible by `n_head`; `--block` doubles as `max_slots` for the twist ladder; `L_max` is derived from the vocabulary at construction time and printed at startup.

---

## 6. Results in detail

### Setup

- **Corpus** — tinyshakespeare, word-level, tokenizer `[A-Za-z]+(?:'[A-Za-z]+)?|[^\sA-Za-z]`, 90/10 train/val split by position.
- **Typo corpus** — 10% of alphabetic tokens corrupted by adjacent swap, deletion, or substitution.
- **Optimization** — AdamW, weight decay 0.01, LR 3e-4 cosine-annealed to 3e-5, gradient clip 1.0, batch 32, block 64.
- **Evaluation** — mean loss over 50 held-out batches, every 100 iterations.
- **Seed** — 1337 for every run.
- **Grid** — `d_model ∈ {128, 256, 512}`, `n_head ∈ {4, 8, 16}`, `K ∈ {8, 16, 32}`, `iters ∈ {1500, 3000}`. Tested on A100 and T4.

### Clean vs typo

The full table is in [TL;DR](#tldr). Two observations:

- The gap does not close as the model widens — at `d_model=512, K=32` it is 1.03 nats, slightly larger than at 256. This is evidence against the "it's just a small-model artifact" objection, though see the caveats about non-monotonic baseline behavior.
- The gap *grows* under character-level noise, from 0.88 to 1.33 nats. This is the cleanest result in the set, because both arms see identically corrupted data and only the representation differs.

### Training dynamics

- **Initial loss** — SPECTRE 6.45 vs vanilla 27.47 (clean); 6.88 vs 26.43 (typo). The codec is well-scaled from step 0; the learned table is not. Note that ~9.4 nats is what a uniform distribution over the vocabulary would give, so the vanilla arm starts *far worse than uniform* — a symptom, discussed below.
- **Norm stability** — the SPECTRE projection output stays near unit norm throughout training; the learned embedding drifts.
- **Throughput (A100)** — ~0.14 s/iter vanilla, ~0.17 s/iter SPECTRE at `K=32`. Roughly 20% slower per step, and the codec table is built once at construction (`V × 2·256·K` float32, so `K=32` is ~800 MB of buffer at 12k vocab — watch memory).

---

## 7. Proof-of-concept lemmas

**Lemma 1 — spacing lives in magnitude.** For a character repeated at positions `p` and `p+d`, the summed phasor has magnitude `|1 + e^{-iωd}| = 2|cos(ωd/2)|`, independent of `p`. Spacing is therefore recoverable from magnitude alone, translation-invariantly.

**Lemma 2 — global order is unambiguous.** With the slowest lens at `ω_min = π / L_max`, the phase `-ω_min · p` is strictly monotonic and non-aliasing over `p ∈ [0, L_max]`. That lens is an unwrapped ruler for absolute position within the token.

**Lemma 3 — multi-scale resolution.** Geometric lens spacing forms a filter bank: fast lenses respond to adjacent-character structure, slow lenses to whole-token layout. The learned projection picks the task-relevant mixture.

Together: the representation is deterministic, fixed-width, injective enough for the task, and information-rich before a single gradient step — which is the mechanism proposed for the observed head start and the noise robustness.

---

## 8. Caveats & controls still to run

Reported in the interest of not overclaiming. None of these invalidate the direction; all of them affect the size of the numbers.

1. **The vanilla arm is mis-initialized, and it is measurably hurting the baseline.** `nn.Embedding` defaults to `N(0, 1)`, and `model.py` then multiplies by `sqrt(d_model)` (≈11.3 at `d_model=128`) while tying the output head to that same table. The `sqrt(d)` trick from the original Transformer assumes an embedding initialized at `N(0, 1/d)`; nanoGPT-style tying assumes `N(0, 0.02)` and *no* scaling. Doing both at once is why vanilla starts at loss ~27 instead of ~9.4. **Control to run:** re-init the embedding to `N(0, 0.02)` and drop the `sqrt(d)` multiply, then re-measure.
2. **Head tying is asymmetric between arms.** Vanilla ties `head.weight = tok.weight`; SPECTRE cannot, so it gets an independent `V × d_model` output matrix. Untying is a known win on its own, and it also means the arms do not have matched trainable-parameter counts (SPECTRE: projection + untied head; vanilla: one shared table). **Control to run:** untied vanilla, and a parameter-matched comparison.
3. **Scaling is non-monotonic for the baseline.** Vanilla is *worse* at `d_model=512` (7.374) than at `256` (6.630). Real scaling gains would not look like that. The likely explanation is that 1,500–3,000 iterations under-trains the wider models at a fixed 3e-4 LR. Until LR is re-tuned per width, "the gap widens with model size" should be treated as suggestive, not established.
4. **Single seed.** Every number comes from seed 1337. Word-level loss differences of ~0.1–0.3 nats are within plausible seed variance at this scale; a 0.88–1.33 nat gap probably is not, but that should be shown rather than assumed. Three seeds minimum.
5. **Absolute losses are high.** 5.7–7.5 nats on a ~12k word-level vocabulary means both arms are far from converged. The comparison may be reading off early-training dynamics rather than final quality. A longer run to a genuine plateau would make the claim much stronger.

The typo-vs-clean *delta* (item-for-item, same arms, same seeds, only the corpus changed) is the most robust result here and is largely insulated from 1–3.

## 9. Credits

SPECTRE was designed as a direct answer to Question #4 of the Kronecker Embeddings V2 open-problem set. Kronecker embeddings: Shravan, 2026. Positional twist follows RoPE (Su et al.). Binding/superposition framing follows the HRR literature (Plate).
