# utils/backtest.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Dict, Tuple

import numpy as np

from .predictor import Predictor, GeradorApostasConfig, FilterConfig

# Limites defensivos (mantenha alinhado ao bot.py)
JANELA_MIN, JANELA_MAX = 50, 1000
ALPHA_MIN, ALPHA_MAX   = 0.05, 0.95
BILH_MIN, BILH_MAX     = 1, 20

# Padrões (alinhado ao bot.py revisado)
DEFAULT_JANELA  = 100
DEFAULT_ALPHA   = 0.30
DEFAULT_BILHETS = 5


@dataclass
class ResultadoBacktest:
    total_concursos: int
    bilhetes_por_concurso: int
    dist_acertos: Dict[int, int]   # mapa: acertos -> quantidade de bilhetes (0..15)
    proporcao_11_mais: float       # razão de bilhetes com >= 11 acertos
    parametros_efetivos: Tuple[int, int, float]  # (janela, bilhetes, alpha)
    # Métricas adicionais
    media_por_aposta: float
    desvio_por_aposta: float
    minimo_por_aposta: int
    maximo_por_aposta: int
    minimo_pior_por_concurso: int
    pct_concursos_pior_ge11: float
    pct_concursos_melhor_ge11: float
    pct_concursos_melhor_ge12: float
    proporcao_12_mais: float       # razão de bilhetes com >= 12 acertos


def _acertos(bilhete: List[int], sorteio: Set[int]) -> int:
    return len(set(bilhete) & sorteio)


def _sanitize_historico(historico: List[Set[int]]) -> List[Set[int]]:
    """
    Garante que cada item do histórico seja um set[int] com valores 1..25.
    Itens inválidos são normalizados; lança erro se algum sorteio não tiver 15 dezenas.
    """
    norm: List[Set[int]] = []
    for i, s in enumerate(historico):
        s_norm = set(int(x) for x in s if isinstance(x, int) or str(x).isdigit())
        s_norm = {n for n in s_norm if 1 <= n <= 25}
        if len(s_norm) != 15:
            raise ValueError(f"Sorteio na posição {i} inválido: esperado 15 dezenas válidas, obtido {len(s_norm)}.")
        norm.append(s_norm)
    return norm


def _validar_parametros(
    janela: int | None,
    bilhetes_por_concurso: int | None,
    alpha: float | None
) -> Tuple[int, int, float]:
    """
    Aplica defaults e limites defensivos. Valores fora de faixa são ajustados (clamp).
    """
    j = DEFAULT_JANELA  if janela is None else int(janela)
    b = DEFAULT_BILHETS if bilhetes_por_concurso is None else int(bilhetes_por_concurso)
    a = DEFAULT_ALPHA   if alpha is None else float(alpha)

    # clamp
    j = max(JANELA_MIN, min(JANELA_MAX, j))
    b = max(BILH_MIN,   min(BILH_MAX,   b))
    a = max(ALPHA_MIN,  min(ALPHA_MAX,  a))

    return j, b, a


def executar_backtest(
    historico: List[Set[int]],
    janela: int = DEFAULT_JANELA,
    bilhetes_por_concurso: int = DEFAULT_BILHETS,
    alpha: float = DEFAULT_ALPHA
) -> ResultadoBacktest:
    """
    Backtest rolling:
      Para cada t a partir de `janela`, treina no intervalo [t-janela, t) e avalia no concurso t.
      Usa seed=t para reprodutibilidade na geração.
    Requisitos:
      - 'historico' deve conter sorteios com 15 dezenas válidas (1..25).
      - len(historico) > janela
    """
    # Normalização/validação de parâmetros
    janela, bilhetes_por_concurso, alpha = _validar_parametros(janela, bilhetes_por_concurso, alpha)

    # Sanitização do histórico
    if not historico or len(historico) < 2:
        raise ValueError("Histórico vazio ou insuficiente.")
    historico = _sanitize_historico(historico)

    if len(historico) <= janela:
        raise ValueError(
            f"Histórico insuficiente para a janela especificada: "
            f"len(historico)={len(historico)} <= janela={janela}."
        )

    # Config padrão de produção para o gerador (alinhado ao bot.py)
    filtro = FilterConfig(
        paridade_min=6,
        paridade_max=9,
        col_min=1,
        col_max=4,
        relax_steps=2,
    )

    cfg_base = GeradorApostasConfig(
        janela=janela,
        alpha=alpha,
        filtro=filtro,
        pool_multiplier=3,
    )

    dist: Dict[int, int] = {}
    total_bilhetes = 0
    lista_acertos: List[int] = []
    pior_por_concurso: List[int] = []
    melhor_por_concurso: List[int] = []

    # Loop rolling
    for t in range(janela, len(historico)):
        treino = historico[t - janela: t]
        alvo = historico[t]

        modelo = Predictor(cfg_base)
        modelo.fit(treino, janela=janela)

        # Geração reprodutível por t
        bilhetes = modelo.gerar_apostas(qtd=bilhetes_por_concurso, seed=t)

        acertos_concurso: List[int] = []
        for b in bilhetes:
            a = _acertos(b, alvo)
            acertos_concurso.append(a)
            dist[a] = dist.get(a, 0) + 1
            lista_acertos.append(a)
        total_bilhetes += len(bilhetes)

        pior_por_concurso.append(min(acertos_concurso))
        melhor_por_concurso.append(max(acertos_concurso))

    # Estatísticas agregadas por aposta
    arr = np.array(lista_acertos, dtype=float) if lista_acertos else np.array([], dtype=float)
    media = float(arr.mean()) if arr.size else 0.0
    desvio = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    minimo = int(arr.min()) if arr.size else 0
    maximo = int(arr.max()) if arr.size else 0
    prop_11 = float((arr >= 11).mean()) if arr.size else 0.0
    prop_12 = float((arr >= 12).mean()) if arr.size else 0.0

    # Métricas por concurso (pior/melhor)
    pior_arr = np.array(pior_por_concurso, dtype=float) if pior_por_concurso else np.array([], dtype=float)
    melhor_arr = np.array(melhor_por_concurso, dtype=float) if melhor_por_concurso else np.array([], dtype=float)

    min_pior = int(pior_arr.min()) if pior_arr.size else 0
    pct_pior_ge11 = float((pior_arr >= 11).mean()) if pior_arr.size else 0.0
    pct_melhor_ge11 = float((melhor_arr >= 11).mean()) if melhor_arr.size else 0.0
    pct_melhor_ge12 = float((melhor_arr >= 12).mean()) if melhor_arr.size else 0.0

    # Preenche chaves ausentes (0..15) com zero para facilitar leitura/plot
    for k in range(0, 16):
        dist.setdefault(k, 0)

    # Ordena por acertos desc (15 -> 0)
    dist_ordenada = dict(sorted(dist.items(), key=lambda kv: kv[0], reverse=True))

    return ResultadoBacktest(
        total_concursos=(len(historico) - janela),
        bilhetes_por_concurso=bilhetes_por_concurso,
        dist_acertos=dist_ordenada,
        proporcao_11_mais=prop_11,
        parametros_efetivos=(janela, bilhetes_por_concurso, alpha),
        media_por_aposta=media,
        desvio_por_aposta=desvio,
        minimo_por_aposta=minimo,
        maximo_por_aposta=maximo,
        minimo_pior_por_concurso=min_pior,
        pct_concursos_pior_ge11=pct_pior_ge11,
        pct_concursos_melhor_ge11=pct_melhor_ge11,
        pct_concursos_melhor_ge12=pct_melhor_ge12,
        proporcao_12_mais=prop_12,
    )


def executar_backtest_resumido(
    historico: List[Set[int]],
    janela: int = DEFAULT_JANELA,
    bilhetes_por_concurso: int = DEFAULT_BILHETS,
    alpha: float = DEFAULT_ALPHA
) -> str:
    """
    Wrapper textual para o /backtest.
    Mantém a mesma assinatura que o seu bot.py utiliza: argumentos posicionais nomeados.
    """
    r = executar_backtest(historico, janela, bilhetes_por_concurso, alpha)
    j, b, a = r.parametros_efetivos

    linhas = [
        "Parâmetros:",
        f"  janela={j} | bilhetes_por_concurso={b} | alpha={a:.2f}",
        f"Concursos avaliados: {r.total_concursos}",
        f"Bilhetes por concurso: {r.bilhetes_por_concurso}",
        "",
        "Distribuição de acertos (acertos: quantidade):",
    ]
    for k in range(15, -1, -1):
        linhas.append(f"  {k}: {r.dist_acertos.get(k, 0)}")

    linhas.extend([
        "",
        "Métricas por aposta:",
        f"  Média: {r.media_por_aposta:.3f} | Desvio: {r.desvio_por_aposta:.3f}",
        f"  Mín/Máx: {r.minimo_por_aposta} / {r.maximo_por_aposta}",
        f"  %≥11 por aposta: {r.proporcao_11_mais:.2%}",
        f"  %≥12 por aposta: {r.proporcao_12_mais:.2%}",
        "",
        "Por concurso (entre as apostas do concurso):",
        f"  Mínimo (pior) observado: {r.minimo_pior_por_concurso}",
        f"  % de concursos com pior ≥11: {r.pct_concursos_pior_ge11:.2%}",
        f"  % de concursos com melhor ≥11: {r.pct_concursos_melhor_ge11:.2%}",
        f"  % de concursos com melhor ≥12: {r.pct_concursos_melhor_ge12:.2%}",
    ])
    return "\n".join(linhas)
