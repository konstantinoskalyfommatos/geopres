# GeoPres

[![PyPI](https://img.shields.io/pypi/v/geopres)](https://pypi.org/project/geopres/)
[![Python versions](https://img.shields.io/pypi/pyversions/geopres)](https://pypi.org/project/geopres/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/konstantinoskalyfommatos/geopres/blob/main/LICENSE)

GeoPres compresses high-dimensional text embeddings with a **single linear map**
that is trained to keep pairwise Euclidean distances. It is *post-hoc*: the
embedding model stays frozen, and at inference time the whole "reduction" is one
matrix multiply. The method is described in our [**EACL paper: "On Post-Hoc Dimensionality Reduction of Text Embeddings"**](https://openreview.net/forum?id=vAWo3CIzqp).

This repository features three parts:

1. **`geopres`, a small library** — train a projection on your own embeddings,
   save it, reload it, and use it on new text (no GPU required).
   - [**Install**](#install)
   - [**Quick start (library)**](#Quick-start-(library))
   - [**API summary**](#API-summary)
2. [**Reproducing the paper**](#Reproducing-the-paper): pre-encode 10M C4 passages, train,
   run MTEB, compare against PCA / random projection / random selection /
   truncation / autoencoder baselines, and rebuild the paper's tables and plots.
3. [**Miscellanea**](Miscellanea) like a repository overview, FAQ, and a bibtex citation that you can use for citing the project.

---

## Install

Python 3.10 or newer is required (tested on 3.10, 3.11, 3.12 and 3.13).

### Library (PyPI)

`pip install geopres` installs the [library](https://pypi.org/project/geopres/) only:

```bash
pip install geopres                # library only: torch + sentence-transformers
pip install "geopres[research]"    # + datasets, mteb, matplotlib, ... (paper pipeline)
pip install "geopres[spearman]"    # + torchsort, only for the --spearman loss
```

### From source (development, paper pipeline)

```bash
git clone https://github.com/konstantinoskalyfommatos/geopres
cd geopres

# recommended: uv, exactly how the paper runs were made
uv sync --extra research                     # package + paper dependencies
uv sync --extra research --extra spearman    # ... plus torchsort, for --spearman

# or with plain pip
pip install -e ".[research]"
```

`torchsort` is not installed by default because it has to be compiled against
your CUDA toolchain; see [Common issues](#common-issues).

---

## Quick start (PyPI library)

There are two ways to use `geopres`, depending on whether you want to train your
own projection or use one that is already trained.

### Fitting your own projection (`GeoPres`)

`GeoPres` wraps a frozen embedding model plus a trainable linear map. Train,
save, and reduce new text:

```python
# pip install geopres
from geopres import GeoPres

model = GeoPres("sentence-transformers/all-mpnet-base-v2", dim=32)
model.fit(texts)                       # texts: list[str]
reduced = model.embed(texts)           # numpy array, shape (len(texts), 32)

model.save("geopres-mpnet-32")
```

Reload it later and use it on new text

```python
from geopres import GeoPres

# Backbone + projection
model = GeoPres.load_path("geopres-mpnet-32")  # the backbone id is saved in`config.json`
reduced = model.embed(["a sentence I have never seen before"])
```

**Training on precumputed embeddings (recommended)** Encoding is the expensive part, and
`fit` accepts embeddings you already have, which makes training much cheaper:

```python
import torch
from geopres import GeoPres, encode_coprus

# Get training vectors
encode_corpus(texts, "my_training_embeddings.pt", "sentence-transformers/all-mpnet-base-v2")

# mmap=True maps the file instead of reading it into RAM (optional, slower)
embeddings = torch.load("my_training_embeddings.pt", mmap=True, weights_only=True)  # (n, 768)

# Reuse training embeddings for multiple .fit calls
GeoPres("sentence-transformers/all-mpnet-base-v2", dim=32).fit(embeddings)
GeoPres("sentence-transformers/all-mpnet-base-v2", dim=64).fit(embeddings)
```

### Using an already-fitted projection (`Projection`)

Every trained `GeoPres` model reduces to a single matrix. `Projection` is that
matrix on its own: a `(dim, source_dim)` weight and nothing else. No backbone is
loaded, downloaded or needed.

This is the right tool when your embeddings come from somewhere else — Hugging
Face's [TEI](https://huggingface.co/docs/text-embeddings-inference), a vector
database, a batch job — as long as they were produced by the embedding model the
projection was trained for. "Same model" means the same pooling, the same
normalisation and, for instruction-based models such as Qwen3-Embedding, the
same query/passage prefixes:

```python
from geopres import Projection

projection = Projection.load_path("geopres-mpnet-32")
# or: Projection.from_pretrained("kkalyfo/geopres-mpnet-32")
projection.source_dim, projection.dim                # (768, 32)

# any vector from the same embedding model: TEI, a vector DB, another service
vector = get_embedding(embedding_model_name="sentence-transformers/all-mpnet-base-v2")
reduced = projection.project(vector)                # (32,)

vectors = ...                                       # (1000, 768)
reduced = projection.project(vectors)               # (1000, 32)
```

If you would rather not depend on `geopres` at inference time, the matrix is
just `projection.weight`: one `vectors @ weight.T`.

Trained projections released with the paper live under
[huggingface.co/kkalyfo/geopres-projections](https://huggingface.co/collections/kkalyfo/geopres-projections).

### API summary

| Call | What it does |
| --- | --- |
| `GeoPres(model, dim=32)` | `model` is a sentence-transformers id, a loaded `SentenceTransformer`, or `None` (embeddings only). |
| `.fit(data, ...)` | Train on texts or on an `(n, source_dim)` array/tensor. Returns `self`. |
| `encode_corpus(texts, path, model)` | Encode a corpus once and save it as one float32 tensor, so `fit` can be re-run without re-encoding. |
| `.embed(data, ...)` | Project texts or embeddings to `dim`. Returns `(n, dim)` numpy by default, `(dim,)` for a single vector. |
| `.save(path)` | Writes `config.json` + `projection.safetensors`. |
| `GeoPres.load_path(path)` | Reloads a saved directory (or a bare `.pt` / `.safetensors` weight file). |
| `.to_sentence_transformer()` | Backbone + projection as a `SentenceTransformer`, e.g. for MTEB. |
| `.push_to_hub(repo_id, private=False)` | Upload `config.json`, weights and a model card to the HF Hub. |
| `GeoPres.from_pretrained(repo_id)` | Download and load a projection from the HF Hub. |
| `Projection.load_path(path)` / `.from_pretrained(repo_id)` | The projection alone, for embeddings you already have. `.project(vectors)`, no backbone. `.to_torch()` for the `nn.Module`. |

`fit` defaults follow the paper (10 epochs, AdamW, lr 1e-2, weight decay 0.1,
linear warmup 0.1, early stopping patience 3, batch size 20 000). Batch size is
a hyperparameter, not only a memory setting: the loss compares all pairs inside
a batch, so a smaller batch trains on a different signal.

Data size matters less than batch size. The paper trained on 10M passages
because more data kept helping, but a few hundred thousand text samples are
enough to get a good projection (see [Data](#reproducing-the-paper)).

---

## Reproducing the paper

The pipeline is **precalculate → train → evaluate → collect**.

```bash
# 0) environment
uv sync --extra research

# 1) pre-encode the corpora with a backbone (one-time per backbone, GPU, slow)
uv run geopres-precalculate \
    --model_name jinaai/jina-embeddings-v2-small-en \
    --train --val_test --batch_size 8192

# 2) train the projection
uv run geopres-train \
    --backbone_model jinaai/jina-embeddings-v2-small-en \
    --source_dim 512 --target_dim 32 \
    --train_batch_size 20000 --val_batch_size 20000 \
    --positional_loss_factor 1 --weighted_loss

# 3) evaluate on MTEB (this is also run automatically at the end of step 2)
uv run geopres-eval --backbone_model jinaai/jina-embeddings-v2-small-en --source_dim 512 --target_dim 32

# 4) merge all MTEB caches into one CSV
uv run geopres-compare
```

Everything is also runnable as a module (`python -m geopres.train_model ...`)
if you prefer not to use the console scripts.

**Data.** Step 1 uses `allenai/c4` (`en` split, 10M passages longer than 200
words) for training and `agentlans/sentence-paraphrases` for validation/test.
No dataset is downloaded by the library; only step 1 needs the `datasets`
package.

We used 10M training samples because results kept improving as we added more
data, but this is not a requirement: a few hundred thousand text samples are
already enough for a good projection, and they make the one-time encoding step
much cheaper. Pass `--max_train_examples` to cap the training set.

**Baselines** read the same cached embeddings:

```bash
uv run python -m geopres.baselines.fit_pca --model_name <backbone> --target_dim 32
uv run python -m geopres.baselines.eval_pca --backbone_model <backbone> --target_dim 32
uv run python -m geopres.baselines.train_autoencoder --backbone_model <backbone> --target_dim 32
# ... eval_random_projection, eval_random_selection, eval_truncation
```

**Tables and plots.** `analysis/` contains the scripts that regenerate the
paper's figures, LaTeX tables and bootstrap tests from the cached results. They
are not part of the installed package, e.g.
`uv run python analysis/create_scatter_plot.py`.

### Where files go

Everything lands under `storage/` in the repository:

```
storage/
├── precalculated_embeddings/<dataset>/<backbone>/{train,validation,test}_embeddings.pt
├── trained_models/<backbone>/<model_name>/checkpoint-N/ + best_model.pt
├── trained_autoencoders/<backbone>/<model_name>/...
├── pca/<backbone>/<target_dim>/
└── evaluation_results/{backbone,trained_models,autoencoders}/...
```

`HF_TOKEN` is picked up from the environment by `huggingface_hub` if you need
gated datasets.

---

## Miscellanea

### Repository layout overview

```
geopres/
├── api.py                          # public API: GeoPres (fit / embed / save) and Projection
├── projection.py                   # the linear map W (a bias-free nn.Linear)
├── losses.py                       # positional / angular / Spearman losses
├── encode.py                       # loading a backbone and encoding text
├── config.py                       # paths + dtype helpers
├── train_model.py                  # paper training entry point (HF Trainer)
├── geopres_trainer.py              # HF Trainer subclass with the pairwise loss
├── reduced_sentence_transformer.py # backbone + projection as a SentenceTransformer
├── precalculate_embeddings.py      # one-time encoding of C4 / sentence-paraphrases
├── eval_model.py                   # MTEB + intrinsic eval of a trained projection
├── eval_backbone.py                # MTEB eval of the raw backbone (reference point)
├── eval_utils.py                   # shared MTEB runner and intrinsic metrics
├── compare_results.py              # merges MTEB caches into comparison_results.csv
├── py.typed                        # this package ships type hints
└── baselines/                      # PCA, random projection, random selection, truncation, autoencoder
analysis/                           # paper tables, plots, bootstrap tests (not shipped)
tests/                              # pytest: API, persistence, losses, hub (no GPU, no network)
```

---

### Common issues

#### `torchsort` fails to install or build

`torchsort` is only needed for `--spearman`. It builds from source, so it needs
a CUDA toolchain. If that fails, install a pre-built wheel from the
[torchsort releases](https://github.com/teddykoker/torchsort/releases):

```bash
export TORCHSORT=0.1.10
export TORCH=pt26      # your torch version: pt26, pt25, pt24, pt21, pt20, pt113
export CUDA=cu126      # cpu, cu113, cu117, cu118, cu121, cu124, cu126
export PYTHON=cp312    # cp310, cp311, cp312

uv pip install https://github.com/teddykoker/torchsort/releases/download/v${TORCHSORT}/torchsort-${TORCHSORT}+${TORCH}${CUDA}-${PYTHON}-${PYTHON}-linux_x86_64.whl
```

#### `Alibaba-NLP/gte-multilingual-base` (mGTE) breaks with newer transformers

The paper's environment pins `transformers==4.57.6`. Newer versions break mGTE
at encode time. If you change the pin, re-test mGTE.

#### Out of memory during training

Memory is dominated by the pairwise loss, which compares every pair of
embeddings in a batch, so it grows quadratically with the batch size. The paper
trains with `--train_batch_size 20000`, and that is the configuration the
reported numbers come from: keep it if you want comparable results.

Shrinking the batch is not a free fix. Fewer pairs per step means a different
training signal, so a smaller batch produces a different model, not the same
model on less memory. If 20000 does not fit on your GPU, treat the smaller batch
as a new configuration and re-check the intrinsic validation loss (ideally also
a task or two) instead of assuming the published numbers carry over.

---

### Development

```bash
uv sync --extra research
uv run pytest            # fast, CPU-only, no downloads
```

### Citation

```bibtex
@inproceedings{kalyfommatos2027geopres,
  title     = {On Post-Hoc Dimensionality Reduction of Text Embeddings},
  author    = {Kalyfommatos, Konstantinos and Michail, Andrianos and Huber, Marius
               and Psomas, Bill and Clematide, Simon and Opitz, Juri},
  booktitle = {Proceedings of the 2027 Conference of the European Chapter of the
               Association for Computational Linguistics (EACL 2027)},
  year      = {2027},
  url       = {https://openreview.net/forum?id=vAWo3CIzqp}
}
```

See also [`CITATION.cff`](https://github.com/konstantinoskalyfommatos/geopres/blob/main/CITATION.cff).

## License

MIT — see [`LICENSE`](https://github.com/konstantinoskalyfommatos/geopres/blob/main/LICENSE).
