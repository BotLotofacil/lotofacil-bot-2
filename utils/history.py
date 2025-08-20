# utils/history.py
from __future__ import annotations
import pandas as pd
import re
from pathlib import Path
from typing import List, Set, Iterable, Tuple

# ---------------- utilitários ----------------

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

# ---------------- leitura robusta ----------------

def _read_csv_candidates(path: Path) -> List[pd.DataFrame]:
    """
    Gera duas leituras candidatas:
      1) header=None (arquivo sem cabeçalho)
      2) header=0   (arquivo com cabeçalho)
    Usa engine='python' e sep=None para autodetectar separador.
    """
    dfs: List[pd.DataFrame] = []
    # Sem cabeçalho
    try:
        df0 = pd.read_csv(path, sep=None, engine="python", dtype=str, header=None).fillna("")
        dfs.append(df0)
    except Exception:
        pass
    # Com cabeçalho
    try:
        df1 = pd.read_csv(path, sep=None, engine="python", dtype=str, header=0).fillna("")
        dfs.append(df1)
    except Exception:
        pass
    # Fallback absoluto (vírgula, sem cabeçalho)
    if not dfs:
        dfs.append(pd.read_csv(path, dtype=str, header=None).fillna(""))
    return dfs

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
        # presença repetida de números 1..25
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

def _parse_df(df: pd.DataFrame) -> List[Set[int]]:
    # normaliza nomes (mesmo que sejam índices 0..14)
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    # 1) tenta 15 colunas numéricas
    rows = _try_parse_15_numeric_columns(df)
    if rows:
        return rows

    # 2) tenta coluna única com lista
    rows = _try_parse_blob_column(df)
    return rows

# ---------------- API pública ----------------

def carregar_historico(caminho: str | Path) -> List[Set[int]]:
    p = Path(caminho)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {p}")

    dfs = _read_csv_candidates(p)

    # escolhe a leitura que renderiza MAIS linhas válidas
    best_rows: List[Set[int]] = []
    for cand in dfs:
        rows = _parse_df(cand)
        rows = [s for s in rows if len(s) == 15 and all(1 <= x <= 25 for x in s)]
        if len(rows) > len(best_rows):
            best_rows = rows

    if not best_rows:
        raise ValueError("Não foi possível interpretar o histórico: nenhuma linha válida com 15 dezenas.")
    return best_rows

def ultimos_n_concursos(historico: List[Set[int]], n: int) -> List[Set[int]]:
    if n <= 0:
        return []
    return historico[-n:] if len(historico) > n else historico

