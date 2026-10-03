"""Efficiency: inference sequences/s and p50 / p95 latency (CUDA events, 50 warm-up + 200 timed
iterations), peak memory (``torch.cuda.max_memory_allocated``, reset per measurement) and training
tokens/s, as functions of sequence length and batch size (spec §6)."""

import time
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import torch

from mambacls.data.collate import Batch


def _synthetic_batch(batch_size, seq_len, vocab_size, n_classes, device):
    ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    lengths = torch.full((batch_size,), seq_len, device=device, dtype=torch.long)
    labels = torch.randint(0, n_classes, (batch_size,), device=device)
    return Batch(ids, lengths, labels, attention_mask=torch.ones_like(ids))


def _timer(device):
    if device.type == "cuda":
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        def run(fn):
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end)
    else:
        def run(fn):
            t = time.perf_counter()
            fn()
            return (time.perf_counter() - t) * 1e3
    return run


def benchmark_efficiency(model, seq_lens: Iterable[int], batch_sizes: Iterable[int] = (1, 32), vocab_size: int = 50280,
                         n_classes: int = 2, warmup: int = 50, iters: int = 200, train: bool = False,
                         autocast_dtype=torch.bfloat16, device=None) -> pd.DataFrame:
    device = device or next(model.parameters()).device
    timer = _timer(device)
    rows: List[dict] = []
    for L in seq_lens:
        for bs in batch_sizes:
            batch = _synthetic_batch(bs, L, vocab_size, n_classes, device)
            use_ac = autocast_dtype is not None and device.type == "cuda"

            def step():
                with torch.autocast("cuda", dtype=autocast_dtype) if use_ac else torch.autocast("cpu", enabled=False):
                    if train:
                        model.train()
                        out = model(batch)
                        out.loss.backward()
                        model.zero_grad(set_to_none=True)
                    else:
                        model.eval()
                        with torch.no_grad():
                            model(batch)

            row = {"seq_len": L, "batch_size": bs, "mode": "train" if train else "inference", "device": str(device)}
            try:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)
                for _ in range(warmup):
                    step()
                times = np.array([timer(step) for _ in range(iters)])
                row.update({
                    "latency_p50_ms": float(np.percentile(times, 50)),
                    "latency_p95_ms": float(np.percentile(times, 95)),
                    "seqs_per_s": bs / (times.mean() / 1e3),
                    "tokens_per_s": bs * L / (times.mean() / 1e3),
                    "peak_mem_mb": torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else float("nan"),
                    "oom": False,
                })
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                row.update({"oom": True})
            rows.append(row)
    return pd.DataFrame(rows)
