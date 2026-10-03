"""Dataset registry (spec §4). Sizes are the commonly published ones; ingest recomputes and logs them."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    hf_id: Optional[str]
    text_fields: Tuple[str, ...]
    label_field: str
    n_classes: int
    config: Optional[str] = None
    revision: Optional[str] = None  # pin a commit SHA in conf/data/*.yaml
    train_split: str = "train"
    val_split: Optional[str] = None  # None: stratified 10% split of train
    test_split: Optional[str] = "test"
    multilabel: bool = False
    cv_folds: Optional[int] = None  # small datasets evaluated with repeated k-fold CV
    byte_level: bool = False  # LRA-style tokenizer-free input
    length_regime: str = "short"
    role: str = ""
    published_sizes: Dict[str, int] = field(default_factory=dict)


DATASETS: Dict[str, DatasetSpec] = {
    "sst2": DatasetSpec(
        "sst2", "stanfordnlp/sst2", ("sentence",), "label", 2, val_split=None, test_split="validation",
        role="short; GLUE comparability with Galim et al.", published_sizes={"train": 67349, "validation": 872},
    ),
    "trec": DatasetSpec(
        "trec", "CogComp/trec", ("text",), "coarse_label", 6, role="very short, low-resource",
        published_sizes={"train": 5452, "test": 500},
    ),
    "trec_fine": DatasetSpec("trec_fine", "CogComp/trec", ("text",), "fine_label", 50, role="very short, 50 classes"),
    "banking77": DatasetSpec(
        "banking77", "PolyAI/banking77", ("text",), "label", 77, role="many-class; head and calibration stress",
        published_sizes={"train": 10003, "test": 3080},
    ),
    "ag_news": DatasetSpec(
        "ag_news", "fancyzhx/ag_news", ("text",), "label", 4, length_regime="medium",
        published_sizes={"train": 120000, "test": 7600},
    ),
    "imdb": DatasetSpec(
        "imdb", "stanfordnlp/imdb", ("text",), "label", 2, length_regime="medium",
        published_sizes={"train": 25000, "test": 25000},
    ),
    "yelp": DatasetSpec(
        "yelp", "fancyzhx/yelp_polarity", ("text",), "label", 2, length_regime="medium",
        role="throughput and scaling curves", published_sizes={"train": 560000, "test": 38000},
    ),
    "dbpedia": DatasetSpec(
        "dbpedia", "fancyzhx/dbpedia_14", ("title", "content"), "label", 14, length_regime="medium",
        published_sizes={"train": 560000, "test": 70000},
    ),
    "arxiv_cls": DatasetSpec(
        "arxiv_cls", "ccdv/arxiv-classification", ("text",), "label", 11, val_split="validation",
        length_regime="long", role="primary long-context test",
    ),
    "hyperpartisan": DatasetSpec(
        "hyperpartisan", "SemEvalWorkshop/hyperpartisan_news_detection", ("title", "text"), "hyperpartisan", 2,
        config="byarticle", test_split=None, cv_folds=5, length_regime="long",
        role="small long-document; variance stress test", published_sizes={"train": 645},
    ),
    "lra_text": DatasetSpec(
        "lra_text", "stanfordnlp/imdb", ("text",), "label", 2, byte_level=True, length_regime="long",
        role="LRA Text (byte-level IMDB, 4K); bidirectional and long-context arms only",
    ),
    "lra_listops": DatasetSpec(
        "lra_listops", None, ("text",), "label", 10, byte_level=False, length_regime="long",
        role="LRA ListOps (synthetic, generated locally)",
    ),
}


def get_spec(name: str) -> DatasetSpec:
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    return DATASETS[name]
