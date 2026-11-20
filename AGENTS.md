# AGENTS.md — GeoPres

GeoPres trains a linear projection that compresses high-dimensional sentence
embeddings while preserving pairwise Euclidean distances, then evaluates it on
MTEB against PCA / random projection / random selection / truncation /
autoencoder baselines. Single-author thesis research code.

The repository is both an installable package (`geopres/`) and the paper's
experiment pipeline.

## Quick start

```bash
uv sync --extra research        # package + paper dependencies
uv run pytest                   # fast, CPU-only, no network
```

All artefacts go to `<repository>/storage`; the paths are defined in
`geopres/config.py`. There is no `.env` file to copy and no `PYTHONPATH` setup any
more.

## Layout

- `geopres/api.py` — public API (`GeoPres.fit` / `.embed` / `.save` /
  `GeoPres.load_path` / `.push_to_hub` / `GeoPres.from_pretrained`). Keep it free
  of `mteb`, `datasets`, `torchsort` and `huggingface_hub` module-level imports:
  `import geopres` must stay light and work with core dependencies only.
- `geopres/projection.py` — the linear map `W`, a bias-free `nn.Linear`, so
  existing `best_model.pt` files load unchanged.
- `geopres/losses.py` — positional / angular / Spearman losses and
  `geopres_loss`, the single place where the loss is combined. `torchsort` is
  imported lazily inside the Spearman functions.
- `geopres/encode.py` — backbone loading and text encoding;
  `sentence_transformers` is imported lazily.
- `geopres/geopres_trainer.py` — Hugging Face `Trainer` used by the paper
  pipeline (checkpoints, early stopping).
- `geopres/train_model.py`, `precalculate_embeddings.py`, `eval_model.py`,
  `eval_backbone.py` — CLI entry points (`geopres-train`,
  `geopres-precalculate`, `geopres-eval`, `geopres-eval-backbone`). Each exposes
  `main()` and also runs with `python -m`.
- `geopres/compare_results.py` — merges the MTEB caches into
  `comparison_results.csv` (entry point `geopres-compare`).
- `geopres/baselines/` — the baseline methods.
- `analysis/` — paper tables, plots and bootstrap tests. Not part of the
  package, needs the cached results in `storage/`.

## Imports

Always use absolute package imports (`from geopres.config import ...`). Never
rely on `PYTHONPATH` or bare imports (`from config import ...`).

## Workflow: precalculate → train → evaluate

1. **Precalculate embeddings** (one time per backbone; GPU-heavy):
   `geopres-precalculate --model_name <hf-id> --train --val_test --batch_size 8192`.
   Writes `$STORAGE_PATH/precalculated_embeddings/{c4|sentence-paraphrases}/<backbone>/{train,validation,test}_embeddings.pt`.
   Encodes in chunks every `--encode_every` (default 100k), skips existing
   chunks, then concatenates and deletes them.
2. **Train**:
   `geopres-train --backbone_model ... --source_dim 512 --target_dim 32 --train_batch_size 20000 --val_batch_size 20000 --positional_loss_factor 1 --weighted_loss`.
   Defaults: AdamW lr=1e-2, weight decay 0.1, 10 epochs, early stopping
   patience 3, linear schedule with 0.1 warmup. `--spearman` swaps in the
   Spearman loss, `--positional_loss_factor < 1` mixes in the angular loss,
   `--resume_from_checkpoint` resumes from the highest `checkpoint-N`.
3. **Evaluate**: `geopres-eval` (trained projection) or `geopres-eval-backbone`
   (raw backbone). Both run MTEB through `eval_utils.evaluate_mteb`, which
   caches under `$EVALUATION_RESULTS_PATH`.
4. **Collect**: `geopres-compare` writes `comparison_results.csv` into
   `$EVALUATION_RESULTS_PATH`. The scripts in `analysis/` turn that into the
   paper's tables and figures.

## Pinned / fragile dependencies

- `transformers==4.57.6` — upgrading breaks
  `Alibaba-NLP/gte-multilingual-base` (mGTE) at encode time. Do not bump
  without retesting.
- `sentence-transformers==5.3.0` — the version behind every paper number.
- `mteb==2.12.26` — the version behind the result caches.
  `eval_utils.evaluate_sts` is tested against it; re-run the evaluation if you
  bump it.
- `torchsort` is an optional extra (`--extra spearman`). It builds from source
  and needs a CUDA toolchain; pre-built wheels are linked from `README.md`
  ("Common issues").

## Devices

`config.default_device()` returns `cuda` when a GPU is visible, otherwise
`cpu`. Scripts log a warning instead of failing on CPU; the paper configuration
still needs a GPU. The library API works on CPU.

## Tests

`tests/` holds small pytest tests for the API, the save/load round trip, the
Hub helpers and the losses. They must stay CPU-only, network-free and fast: no
model downloads, no MTEB runs, no real `push_to_hub` (mock
`huggingface_hub.snapshot_download` instead).

## Conventions

- `torch.manual_seed(42)` at module top of `eval_utils.py` and
  `geopres_trainer.py`.
- `TOKENIZERS_PARALLELISM=false` in the training and precalculate scripts.
- `HF_HUB_DOWNLOAD_TIMEOUT=300` in `precalculate_embeddings.py`.
- Model names contain the backbone, target dim, batch size and method suffix;
  adding a suffix requires updating `_get_extrinsic_model_names` in
  `analysis/calculate_intrinsic_correlations.py`.

## What not to do

- Do not bump `transformers` or `sentence-transformers` without checking mGTE.
- Do not import `mteb`, `datasets` or `torchsort` at module level in library
  modules (it breaks core-only installs).
- Do not re-introduce `.env` / `python-dotenv` or hard-coded `cuda`.
- Do not `pip install` globally — use `uv`.
