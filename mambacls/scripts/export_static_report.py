"""Export every dashboard page to self-contained HTML (one file per page + an index).

    python scripts/export_static_report.py --store results/store --artifacts results/artifacts.zarr --out report/
"""

import argparse
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mambacls.store.results import ResultsStore  # noqa: E402
from mambacls.viz.pages import PAGES  # noqa: E402


def export(store_dir, artifacts_path, out_dir, filters=None, inline_js: bool = False):
    store = ResultsStore(store_dir)
    arts = None
    if artifacts_path and Path(artifacts_path).exists():
        from mambacls.store.artifacts import ArtifactStore

        arts = ArtifactStore(artifacts_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    links = []
    for name, fn in PAGES.items():
        figs = fn(store, arts, filters or {})
        slug = name.lower().replace(" ", "_")
        parts = [f"<h1>{html.escape(name)}</h1>"]
        for i, (fid, fig) in enumerate(figs):
            parts.append(fig.to_html(full_html=False, include_plotlyjs=(True if inline_js else "cdn") if i == 0 else False, div_id=fid))
        (out / f"{slug}.html").write_text(
            "<!doctype html><html><head><meta charset='utf-8'><title>%s</title></head>"
            "<body style='font-family:Inter,Helvetica,Arial,sans-serif;max-width:1400px;margin:auto;background:#fcfcfb'>%s</body></html>"
            % (html.escape(name), "\n".join(parts)))
        links.append(f"<li><a href='{slug}.html'>{html.escape(name)}</a> ({len(figs)} figures)</li>")
    (out / "index.html").write_text("<!doctype html><html><head><meta charset='utf-8'><title>mambacls report</title></head>"
                                    "<body style='font-family:Inter,Helvetica,Arial,sans-serif'><h1>mambacls report</h1><ul>"
                                    + "".join(links) + "</ul></body></html>")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="results/store")
    ap.add_argument("--artifacts", default="results/artifacts.zarr")
    ap.add_argument("--out", default="report")
    ap.add_argument("--inline-js", action="store_true", help="embed plotly.js (offline viewing)")
    a = ap.parse_args()
    print(export(a.store, a.artifacts, a.out, inline_js=a.inline_js))


if __name__ == "__main__":
    main()
