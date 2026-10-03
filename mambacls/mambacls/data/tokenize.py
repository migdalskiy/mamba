"""Tokenisation with explicit truncation policies (spec §6: always report the policy used)."""

from typing import List, Optional

TRUNCATION_POLICIES = ("head", "tail", "head_tail", "none")


class ByteTokenizer:
    """Tokenizer-free UTF-8 byte input (LRA Text). ids 0..255 are bytes, 256 = pad, 257 = eos."""

    pad_token_id = 256
    eos_token_id = 257
    vocab_size = 258
    name_or_path = "bytes"

    def __call__(self, texts, add_special_tokens=False, **kwargs):
        if isinstance(texts, str):
            return {"input_ids": list(texts.encode("utf-8"))}
        return {"input_ids": [list(t.encode("utf-8")) for t in texts]}


class WhitespaceTokenizer:
    """Token-per-word fallback for synthetic tasks and offline tests."""

    pad_token_id = 0
    eos_token_id = 1
    name_or_path = "whitespace"

    def __init__(self, vocab: Optional[List[str]] = None, vocab_size: int = 50280):
        self.vocab = {w: i + 2 for i, w in enumerate(vocab or [])}
        self.vocab_size = vocab_size

    def _id(self, w):
        if w in self.vocab:
            return self.vocab[w]
        return 2 + (sum(ord(c) * 31 ** i for i, c in enumerate(w)) % (self.vocab_size - 2))

    def __call__(self, texts, add_special_tokens=False, **kwargs):
        if isinstance(texts, str):
            return {"input_ids": [self._id(w) for w in texts.split()]}
        return {"input_ids": [[self._id(w) for w in t.split()] for t in texts]}


def truncate(ids: List[int], max_len: Optional[int], policy: str = "head") -> List[int]:
    """head: keep the first max_len tokens; tail: the last; head_tail: half from each end
    (the usual DeBERTa long-document policy)."""
    if policy not in TRUNCATION_POLICIES:
        raise ValueError(f"truncation must be one of {TRUNCATION_POLICIES}")
    if max_len is None or policy == "none" or len(ids) <= max_len:
        return ids
    if policy == "head":
        return ids[:max_len]
    if policy == "tail":
        return ids[-max_len:]
    head = max_len // 2
    return ids[:head] + ids[len(ids) - (max_len - head):]


def load_tokenizer(name: str):
    if name == "bytes":
        return ByteTokenizer()
    if name == "whitespace":
        return WhitespaceTokenizer()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def tokenize_dataset(ds, tokenizer, max_len: Optional[int], truncation: str = "head", append_eos: bool = False,
                     reserve: int = 0, num_proc: Optional[int] = None):
    """Adds input_ids, n_tokens (before truncation), length, truncated, n_words.

    reserve: positions kept free for tokens added later (an appended readout token, prompt
    prefixes), so the backbone never sees more than max_len positions.
    """
    budget = None if max_len is None else max_len - reserve - int(append_eos)
    eos = getattr(tokenizer, "eos_token_id", None)

    def fn(batch):
        enc = tokenizer(batch["text"], add_special_tokens=False)["input_ids"]
        out = {"input_ids": [], "n_tokens": [], "length": [], "truncated": [], "n_words": []}
        for text, ids in zip(batch["text"], enc):
            t = truncate(list(ids), budget, truncation)
            if append_eos:
                t = t + [eos]
            out["input_ids"].append(t)
            out["n_tokens"].append(len(ids))
            out["length"].append(len(t))
            out["truncated"].append(budget is not None and len(ids) > budget)
            out["n_words"].append(len(text.split()))
        return out

    return ds.map(fn, batched=True, num_proc=num_proc)
