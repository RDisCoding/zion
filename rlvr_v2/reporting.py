"""Shared helpers for the paper outputs: booktabs LaTeX tables, JSON export and matplotlib styling."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# Categorical slots in fixed order (identity colours never cycle), an ordinal blue ramp for p_s bins,
# and recessive chrome colours; see the dataviz palette reference.
PALETTE = {
    "series": ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"),
    "ordinal3": ("#86b6ef", "#2a78d6", "#104281"),
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "surface": "#ffffff",
}
ARM_COLORS = {
    "random": PALETTE["series"][0],
    "variance": PALETTE["series"][1],
    "disagreement": PALETTE["series"][2],
    "ps_band": PALETTE["series"][3],
    "learned": PALETTE["series"][4],
    "repeat_one": PALETTE["series"][5],
}

_TEX_SPECIAL = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "\\": r"\textbackslash{}",
                "<": r"\textless{}", ">": r"\textgreater{}", "|": r"\textbar{}"}


class Raw(str):
    """A table cell / caption that is already valid LaTeX (e.g. ``Raw("$p_s$")``) and must not be escaped."""


def escape_tex(s: Any) -> str:
    """Escape LaTeX special characters in a plain-text cell (feature names such as ``p_s``); ``Raw`` passes through."""
    if isinstance(s, Raw):
        return str(s)
    return "".join(_TEX_SPECIAL.get(ch, ch) for ch in str(s))


def is_missing(v: Any) -> bool:
    if v is None:
        return True
    try:
        return bool(math.isnan(float(v)))
    except (TypeError, ValueError):
        return False


def fmt_num(v: Any, nd: int = 3, signed: bool = False) -> str:
    """Number -> fixed-point string; missing -> an en dash."""
    if is_missing(v):
        return "--"
    return f"{float(v):+.{nd}f}" if signed else f"{float(v):.{nd}f}"


def fmt_ci(lo: Any, hi: Any, nd: int = 3) -> str:
    if is_missing(lo) or is_missing(hi):
        return "--"
    return f"[{float(lo):.{nd}f}, {float(hi):.{nd}f}]"


def fmt_p(p: Any) -> str:
    if is_missing(p):
        return "--"
    p = float(p)
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.4f}" if p < 0.01 else f"{p:.3f}"


def tex_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], colspec: str | None = None,
              caption: str | None = None, label: str | None = None, path: str | Path | None = None,
              escape: bool = True) -> str:
    """Render a booktabs table (``\\toprule``/``\\midrule``/``\\bottomrule``). Cells, headers and the caption
    are escaped unless they are ``Raw`` (or ``escape=False``); numbers should be pre-formatted with
    ``fmt_num``/``fmt_p``. With ``caption`` or ``label`` the tabular is wrapped in a ``table`` environment.
    Writes to ``path`` when given."""
    esc = escape_tex if escape else (lambda s: str(s))
    ncol = len(headers)
    spec = colspec or ("l" + "r" * (ncol - 1))
    lines = []
    wrap = caption is not None or label is not None
    if wrap:
        lines += ["\\begin{table}[t]", "\\centering", "\\small"]
    lines.append(f"\\begin{{tabular}}{{{spec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(esc(h) for h in headers) + " \\\\")
    lines.append("\\midrule")
    for row in rows:
        cells = [esc(c) for c in row]
        if len(cells) != ncol:
            raise ValueError(f"row has {len(cells)} cells, expected {ncol}: {row}")
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if wrap:
        if caption:
            lines.append(f"\\caption{{{esc(caption)}}}")
        if label:
            lines.append(f"\\label{{{label}}}")
        lines.append("\\end{table}")
    tex = "\n".join(lines) + "\n"
    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(tex, encoding="utf-8")
    return tex


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays, tuples, Paths and NaN (-> None) for ``json.dump``."""
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(obj), fh, indent=1)


def setup_matplotlib():
    """Headless matplotlib with recessive chrome (hairline solid grid, no top/right spines, 2px lines).
    Returns the ``pyplot`` module, or None when matplotlib is not installed."""
    try:
        import matplotlib
    except ImportError:  # pragma: no cover - optional dependency
        return None
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": PALETTE["axis"],
        "axes.labelcolor": PALETTE["ink2"],
        "axes.titlecolor": PALETTE["ink"],
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": PALETTE["muted"],
        "ytick.color": PALETTE["muted"],
        "xtick.labelcolor": PALETTE["ink2"],
        "ytick.labelcolor": PALETTE["ink2"],
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })
    return plt
