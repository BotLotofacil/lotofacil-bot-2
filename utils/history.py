# utils/history.py
from __future__ import annotations
import pandas as pd
import re
from pathlib import Path
from typing import List, Set, Iterable

def _only_int_1_25(x: str) -> int | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    if s.isdigit():
        v = int(s)
    else:
        s2 = re.sub(r"\D", "", s)
        if not s2.isdigit():
            return None
        v = int(s2)
    return v if 1 <= v <= 25 else None

def _row_to_set(nums: Iterable[int]) -> Set[int]:
    seen = set()
    out: List[int] = []
    for n in nums:
        if n is None:
            continue
        if n not in seen and 1 <= n <= 25:
            seen.add(n)
            out.append(n)
        if len(out) == 15:
            break
    return set(out)

def _parse_line_blob(txt: str) -> Set[int]:
    parts = re.split(r"[,\s;]+", str(txt or "").strip())
    nums: List[int] = []
    for p in parts:
        v = _only_int_1_25(p)
        if v is not None:
            nums.append(v)
    return _row_to_set(nums)

def _infer_sep_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python", dtype=str).fillna("")

def _try_parse_15_numeric_columns(df: pd.DataFrame) -> List[Set[int]]:
    if df.empty:
        return []
    candidate_cols = []
    sample_n = min(50, len(df))
    for col in df.columns:
        vals = df[col].astype(str).head(sample_n).tolist()
        valid = sum(1 for v in vals if _only_int_1_25(v) is not None)
        ratio = valid / max(1, len(vals))
        if ratio >= 0.70:
            candidate_cols.append(col)
    if len(candidate_cols) < 15:
        return []
    cols15 = candidate_cols[:15]
    rows: List[Set[int]] = []
    for _, r in df[cols15].iterrows():
        nums: List[int] = []
        for v in r.values:
            iv = _only_int_1_25(v)
            if iv is not None:
                nums.append(iv)
        s = _row_to_set(nums)
        if len(s) == 15:
            rows.append(s)
    return rows

def _try_parse_blob_column(df: pd.DataFrame) -> List[Set[int]]:
    if df.empty:
        return []
    rows: List[Set[int]] = []
    chosen = None
    for col in df.columns:
        sample = " ".join(df[col].astype(str).head(3).tolist())
        hits = re.findall(r"\b0?([1-9]|1[0-9]|2[0-5])\b", sample)
        if len(hits) >= 5:
            chosen = col
            break
    if chosen is None:
        return []
    for v in df[chosen].astype(str):
        s = _parse_line_blob(v)
        if len(s) == 15:
            rows.append(s)
    return rows

def carregar_historico(caminho: str | Path) -> List[Set[int]]:
    p = Path(caminho)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {p}")
    try:
        df = _infer_sep_read_csv(p)
    except Exception:
        df = pd.read_csv(p, dtype=str).fillna("")
    df.columns = [str(c).strip().lower() for c in df.columns]
    rows = _try_parse_15_numeric_columns(df)
    if not rows:
        rows = _try_parse_blob_column(df)
    rows = [s for s in rows if len(s) == 15 and all(1 <= x <= 25 for x in s)]
    if not rows:
        raise ValueError("Não foi possível interpretar o histórico: nenhuma linha válida com 15 dezenas.")
    return rows

def ultimos_n_concursos(historico: List[Set[int]], n: int) -> List[Set[int]]:
    if n <= 0:
        return []
    return historico[-n:] if len(historico) > n else historico

    if n <= 0:
        return []
    return historico[-n:] if len(historico) > n else historico
