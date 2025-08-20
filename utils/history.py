# utils/history.py
from __future__ import annotations
import pandas as pd
import re
from pathlib import Path
from typing import List, Set, Iterable

def _row_to_set(nums: Iterable[int]) -> Set[int]:
    s = []
    for x in nums:
        try:
            xi = int(x)
            if 1 <= xi <= 25:
                s.append(xi)
        except Exception:
            continue
    # mantém no máximo 15 elementos, remove duplicados preservando ordem
    seen = set()
    out = []
    for v in s:
        if v not in seen:
            seen.add(v)
            out.append(v)
        if len(out) == 15:
            break
    return set(out)

def _parse_line_blob(txt: str) -> Set[int]:
    """
    Aceita linhas no formato:
      - "01 02 03 04 ...", "1;2;3;...", "1, 2, 3, ..."
      - tolera zeros à esquerda e separadores mistos
    """
    parts = re.split(r"[,\s;]+", str(txt or "").strip())
    nums = []
    for p in parts:
        p = str(p).strip()
        if not p:
            continue
        if p.isdigit():
            nums.append(int(p))
            continue
        # remove caracteres não numéricos e tenta de novo
        p2 = re.sub(r"\D", "", p)
        if p2.isdigit():
            nums.append(int(p2))
    return _row_to_set(nums)

def carregar_historico(caminho: str | Path) -> List[Set[int]]:
    """
    Lê data/history.csv e retorna uma lista de conjuntos de 15 dezenas por concurso.
    Formatos aceitos:
      1) 15 colunas numéricas (ex.: d1..d15, n1..n15, d01..d15)
      2) Uma coluna com string de dezenas separadas por espaço, vírgula ou ponto e vírgula.
    """
    p = Path(caminho)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {p}")

    # Lê tudo como string para tolerar zeros à esquerda e valores não limpos
    df = pd.read_csv(p, dtype=str).fillna("")
    if df.empty:
        raise ValueError("Arquivo de histórico está vazio.")

    # Normaliza nomes de colunas
    cols_lower = [str(c).strip().lower() for c in df.columns]
    df.columns = cols_lower

    # Tentativa 1: detectar 15 colunas de dezenas
    # Padrões comuns: d1..d15, n1..n15, d01..d15, n01..n15
    candidatos = []
    padroes = [
        [f"d{i}" for i in range(1, 16)],
        [f"n{i}" for i in range(1, 16)],
        [f"d{i:02d}" for i in range(1, 16)],
        [f"n{i:02d}" for i in range(1, 16)],
    ]
    for cand in padroes:
        if all(c in df.columns for c in cand):
            candidatos = cand
            break

    rows: List[Set[int]] = []

    if candidatos:
        sub = df[candidatos]
        for _, r in sub.iterrows():
            nums = []
            for v in r.values:
                v = str(v).strip()
                if v.isdigit():
                    nums.append(int(v))
                else:
                    v2 = re.sub(r"\D", "", v)
                    if v2.isdigit():
                        nums.append(int(v2))
            s = _row_to_set(nums)
            if len(s) == 15:
                rows.append(s)
    else:
        # Tentativa 2: procurar uma coluna "blob" com as 15 dezenas
        # Escolhe a primeira coluna que aparente conter listas
        escolhido = None
        for c in df.columns:
            sample = " ".join(df[c].astype(str).head(3).tolist())
            # Heurística simples: presença de números de 1..25
            if re.search(r"\b0?([1-9]|1[0-9]|2[0-5])\b", sample):
                escolhido = c
                break
        if not escolhido:
            raise ValueError("Não encontrei 15 colunas de dezenas nem uma coluna com lista de dezenas.")
        for v in df[escolhido].astype(str):
            s = _parse_line_blob(v)
            if len(s) == 15:
                rows.append(s)

    # Sanitização final
    rows = [s for s in rows if len(s) == 15 and all(1 <= x <= 25 for x in s)]
    if not rows:
        raise ValueError("Não foi possível interpretar o histórico: nenhuma linha válida com 15 dezenas.")
    return rows

def ultimos_n_concursos(historico: List[Set[int]], n: int) -> List[Set[int]]:
    if n <= 0:
        return []
    return historico[-n:] if len(historico) > n else historico
