# mambacls: Mamba-2 / Mamba-3 sequence classifiers and comparison workbench

An implementation of the tech spec *"Mamba-2 / Mamba-3 Sequence Classifiers: Method Catalog and an
End-to-End Comparison Workbench"*. It is built directly on `mamba_ssm` (this repository) and does
not use Hugging Face classification classes. A `MixerModel` backbone without its LM head is
combined with a pooler, a classification head and an adaptation method. Every run writes Parquet
rows and Zarr tensors that a Plotly/Dash dashboard reads.

```
mambacls/
├── conf/            Hydra configs: backbone/ (14 Mamba IDs + Mamba-1 control + baselines), pooler/, adapter/, data/, experiment/
├── mambacls/
│   ├── data/        ingest (pinned revisions, stratified val split, k-fold), tokenize (truncation policies), collate, stats
│   ├── models/      backbone, mixers (decomposed forwards), poolers, heads, registry, baselines, adapters/
│   ├── train/       loop (bf16 autocast, fp32 master weights), optim, callbacks, distill
│   ├── eval/        metrics, calibration, efficiency, significance, landscape
│   ├── probe/       capture (InternalsRecorder), ssd_reference, hidden_attention, hooks
│   ├── store/       results (Parquet + DuckDB, provenance), artifacts (Zarr)
│   ├── viz/         theme, figures/* (pure Plotly factories), pages, app (Dash)
│   └── experiment.py   one end-to-end run
├── scripts/         run.py, sweep.py, significance_report.py, export_static_report.py, checkpoint_smoke.py, demo_cpu.py
└── tests/           CPU suite + GPU parity tests (skipped without CUDA)
```

## Quick start

```bash
pip install -e ../ --no-build-isolation      # mamba_ssm from source (Mamba-3 needs it)
pip install -e .[dev,umap]

python scripts/run.py backbone=mamba3_siso_893m pooler=latent_query adapter=lora data=banking77 seed=0
python scripts/run.py +experiment=peft_sweep                    # the spec's example config
python scripts/sweep.py peft_sweep --print                      # Phase B grid (1160 runs)
python scripts/sweep.py pooling_sweep --shard 0/8 --run         # Phase A, shard 0 of 8
python scripts/significance_report.py --phase B --baseline last/full
python -m mambacls.viz.app --store results/store --artifacts results/artifacts.zarr
python scripts/export_static_report.py --out report/            # static HTML of every page

python scripts/demo_cpu.py --out demo_results                   # CPU-only demo that fills every dashboard page
```

From Python:

```python
from mambacls.config import load_config
from mambacls.experiment import run_experiment
row = run_experiment(load_config(["backbone=mamba2_780m", "pooler=mean", "adapter=state_offset", "data=imdb"]))
```

## What is implemented

| Spec | Where | Notes |
|---|---|---|
| §3.1 poolers: last, eos_cls, mean, max, attn, latent_query, scalar_mix | `models/poolers.py` | All are masked and pad-invariant (tested). `eos_cls` uses a learned appended embedding or the tokenizer EOS. `scalar_mix` reads `norm_f`-normalised per-layer streams. |
| §3.2 probe / full FT | `adapters/base.py`, `train/loop.py` | Probe with a parameter-free pooler caches features and fits a linear head (plus a per-layer probe and scalar-mix weights). Full FT keeps A_log / dt_bias / D in fp32 and never re-initialises the backbone. |
| §3.3 LoRA, SDLoRA | `adapters/lora.py`, `adapters/sdt.py` | LoRA is a **weight parametrization**, because the fused Mamba-2 kernel reads `out_proj.weight` directly and a forward hook would silently be skipped. It supports merge / unmerge. SDT selects heads (Mamba-2/3) or channels (Mamba) after warm-up; frozen entries are restored after every optimizer step. |
| §3.4 State-offset Tuning (y, h), initial-state (h0), prompt / affix | `adapters/state_offset.py`, `adapters/prompt.py` | Zero-initialised, so step 0 is the pretrained model. The h variant adds y_t += C_tᵀh′ (q_t for Mamba-3). |
| §3.5 bidirectional retrofit: tied_add, tied_gate, untied_lora | `adapters/bidir.py`, `models/backbone.py` | Reverses within each sequence's length or varlen segment. tied_gate at init reproduces the causal model (tested). |
| §3.6 long context: naive, train-at-length, LongMamba | `adapters/longmamba.py`, `experiment._length_eval` | A simplified LongMamba-style port for Mamba-2: calibrate per-head decay budgets at L_train and filter tokens of global heads at longer L. Not the reference code. |
| §3.7 hybrid triple | `conf/experiment/hybrid_triple.yaml` | mamba_ssm's MHA layers also name their projections `in_proj` / `out_proj`, so the same LoRA targets apply. |
| §3.8 KD | `train/distill.py` | CE + τ²·KL; optional hidden-attention alignment loss. |
| §3.9 baselines | `models/baselines.py` | ModernBERT / DeBERTa via `AutoModelForSequenceClassification`; Pythia behind the backbone interface with the same poolers. |
| §4 datasets | `data/registry.py`, `conf/data/` | All nine, plus a local LRA ListOps generator and byte-level LRA Text. Hyperpartisan uses repeated stratified k-fold. |
| §6 metrics and statistics | `eval/` | acc, macro-F1, per-class F1, NLL, ECE (15 equal-mass bins), Brier, temperature scaling; efficiency (CUDA events, peak memory); bootstrap CIs, paired bootstrap, Wilcoxon, Holm, the corrected resampled t-test, and the "Holm p < .05 **and** CI excludes 0" decision rule. |
| §7 visualisation | `viz/` | 47 figure factories covering the §7.1 inventory, including the 3D views. Dash app with 10 pages; static export. |
| §7.2–7.4 internals | `models/mixers.py`, `probe/` | Decomposed forward of each mixer (workaround 1) and fp32 reference states, decay L and hidden attention M (workaround 2). M·x + D·x reconstructs the mixer output for Mamba, Mamba-2, Mamba-3 SISO and MIMO (tested). Mamba-3 matrices are labelled experimental. |
| §8 interfaces, collation contract | as specified | Right padding plus `lengths` by default. Varlen packing gives trailing pad its own dummy segment so `seq_idx` covers every position (issue #783). Left padding is for HF backends only. |
| §9 tests | `tests/` | See below. |

### Execution paths

`MambaBackbone(impl="kernel")` uses the fused kernels and only decomposes a mixer when an adapter or
the recorder needs its internals; the scan then still runs through the kernels.
`impl="reference"` runs everything in plain PyTorch, fp32, on any device. That path is the CPU
execution path, the parity oracle, and the source of probe internals. The scan math is shared with
`mamba_ssm.explain` (MambaLRP), where it is validated against step-by-step recurrences and the
repo's kernel references.

## Tests

```bash
cd mambacls && pytest -q        # 163 CPU tests, ~20 s; GPU parity tests skip without CUDA
UPDATE_SNAPSHOTS=1 pytest tests/test_viz.py   # after intentional figure changes
MAMBACLS_HF_SMOKE=1 pytest tests/test_gpu.py  # + checkpoint smoke (GPU + Hugging Face)
```

What the suite checks:

* **Model correctness:** reference mixers match the validated forward. Pad invariance holds for every pooler, both alone and end to end. Bidirectional reversal respects lengths. tied_gate starts exactly causal. Packed and padded inputs give identical outputs.
* **Adapters:** LoRA merge / unmerge round-trips. State offsets start at zero, and the h offset equals the C projection. SDT selects the right heads and restores the frozen ones. The LongMamba filter is a no-op at or below L_train.
* **Training:** the training smoke runs every adapter on Mamba-2 and Mamba-3 (loss decreases, no NaN), training is deterministic, and runs end to end through the store.
* **Statistics and figures:** metric values agree with sklearn and scipy. Every figure factory has a snapshot test. All dashboard pages render.

The GPU-only tests in `tests/test_gpu.py` (reference vs kernels, decomposed vs fused, varlen
kernels, checkpoint smoke) have **not been run yet**: the development container had no GPU. They
are the M0 gate.

## Status and caveats

* **Not run here:** Phases A–D, the checkpoint loads, and all GPU numbers. The development
  environment had no GPU and no Hugging Face access. Treat every Mamba-3 result as provisional
  until `tests/test_gpu.py` passes on the pinned commit (spec §13).
* **Untested kernel glue:** the kernel glue in the decomposed Mamba-3 path (`mamba3_siso_combined`
  / `mamba3_mimo` with Z=None and no output projection) follows the kernels' documented
  signatures. It is covered only by the GPU tests.
* **Single-reference adaptations:** head-level SDT for Mamba-2/3, the LongMamba port, and the
  bidirectional retrofit for text are this workbench's own adaptations, as the spec notes.
* **W&B mirroring:** `tracking.wandb` is a config flag only; the Parquet/DuckDB store is the
  source of truth.
