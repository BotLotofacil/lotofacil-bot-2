# utils/backtest.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Dict, Tuple

from .predictor import Predictor, GeradorApostasConfig

# Limites defensivos (mantenha alinhado ao bot.py)
JANELA_MIN, JANELA_MAX = 50, 1000
ALPHA_MIN, ALPHA_MAX   = 0.05, 0.95
BILH_MIN, BILH_MAX     = 1, 20

# Padrões (mantenha alinhado ao bot.py)
DEFAULT_JANELA  = 200
DEFAULT_ALPHA   = 0.35
DEFAULT_BILHETS = 5

@dataclass
class ResultadoBacktest:
    total_concursos: int
    bilhetes_por_concurso: int
    dist_acertos: Dict[int, int]   # mapa: acertos -> quantidade de bilhetes
    proporcao_11_mais: float       # razão de bilhetes com >= 11 acertos
    parametros_efetivos: Tuple[int, int, float]  # (janela, bilhetes, alpha)

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
    j = DEFAULT_JANELA if janela is None else int(janela)
    b = DEFAULT_BILHETS if bilhetes_por_concurso is None else int(bilhetes_por_concurso)
    a = DEFAULT_ALPHA if alpha is None else float(alpha)

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
      Usa seed=t para reprodutibilidade.
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

    dist: Dict[int, int] = {}
    total_bilhetes = 0

    # Loop rolling
    for t in range(janela, len(historico)):
        treino = historico[t - janela: t]
        alvo = historico[t]

        cfg = GeradorApostasConfig(janela=janela, alpha=alpha)
        modelo = Predictor(cfg)
        # Treina no histórico da janela
        modelo.fit(treino, janela=janela)

        # Gera bilhetes de forma determinística por t
        bilhetes = modelo.gerar_apostas(qtd=bilhetes_por_concurso, seed=t)
        for b in bilhetes:
            a = _acertos(b, alvo)
            dist[a] = dist.get(a, 0) + 1
        total_bilhetes += len(bilhetes)

    acima_11 = sum(v for k, v in dist.items() if k >= 11)
    proporcao = (acima_11 / total_bilhetes) if total_bilhetes else 0.0

    # Preenche chaves ausentes (0..15) com zero para facilitar leitura/plot
    for k in range(0, 16):
        dist.setdefault(k, 0)

    # Ordena por acertos desc (15 -> 0)
    dist_ordenada = dict(sorted(dist.items(), key=lambda kv: kv[0], reverse=True))

    return ResultadoBacktest(
        total_concursos=(len(historico) - janela),
        bilhetes_por_concurso=bilhetes_por_concurso,
        dist_acertos=dist_ordenada,
        proporcao_11_mais=proporcao,
        parametros_efetivos=(janela, bilhetes_por_concurso, alpha),
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
        "Distribuição de acertos (acertos: quantidade):",
    ]
    # Exibe de 15 até 8 (mais relevantes); abaixo de 8 costuma ser pouco útil, mas já está no dicionário se quiser inspecionar
    for k in range(15, 7, -1):
        linhas.append(f"  {k}: {r.dist_acertos.get(k, 0)}")

    linhas.append(f"Proporção de bilhetes com >=11 acertos: {r.proporcao_11_mais:.2%}")
    return "\n".join(linhas)
