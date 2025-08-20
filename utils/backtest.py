# utils/backtest.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Set, Dict

from .predictor import Predictor, GeradorApostasConfig

@dataclass
class ResultadoBacktest:
    total_concursos: int
    bilhetes_por_concurso: int
    dist_acertos: Dict[int, int]   # mapa: acertos -> quantidade de bilhetes
    proporcao_11_mais: float       # razão de bilhetes com >= 11 acertos

def _acertos(bilhete: List[int], sorteio: Set[int]) -> int:
    return len(set(bilhete) & sorteio)

def executar_backtest(
    historico: List[Set[int]],
    janela: int = 200,
    bilhetes_por_concurso: int = 5,
    alpha: float = 0.35
) -> ResultadoBacktest:
    """
    Backtest rolling: para cada t a partir de `janela`, treina no intervalo [t-janela, t)
    e avalia no concurso t. Usa seed=t para reprodutibilidade.
    """
    if len(historico) <= janela:
        raise ValueError("Histórico insuficiente para a janela especificada.")

    dist: Dict[int, int] = {}
    total_bilhetes = 0

    for t in range(janela, len(historico)):
        treino = historico[t - janela: t]
        alvo = historico[t]

        cfg = GeradorApostasConfig(janela=janela, alpha=alpha)
        modelo = Predictor(cfg)
        modelo.fit(treino, janela=janela)

        bilhetes = modelo.gerar_apostas(qtd=bilhetes_por_concurso, seed=t)
        for b in bilhetes:
            a = _acertos(b, alvo)
            dist[a] = dist.get(a, 0) + 1
        total_bilhetes += len(bilhetes)

    acima_11 = sum(v for k, v in dist.items() if k >= 11)
    proporcao = (acima_11 / total_bilhetes) if total_bilhetes else 0.0

    return ResultadoBacktest(
        total_concursos=(len(historico) - janela),
        bilhetes_por_concurso=bilhetes_por_concurso,
        dist_acertos=dict(sorted(dist.items(), reverse=True)),
        proporcao_11_mais=proporcao
    )

def executar_backtest_resumido(
    historico: List[Set[int]],
    janela: int = 200,
    bilhetes_por_concurso: int = 5,
    alpha: float = 0.35
) -> str:
    r = executar_backtest(historico, janela, bilhetes_por_concurso, alpha)
    linhas = [
        f"Concursos avaliados: {r.total_concursos}",
        f"Bilhetes por concurso: {r.bilhetes_por_concurso}",
        "Distribuição de acertos (acertos: quantidade):",
    ]
    for k in sorted(r.dist_acertos.keys(), reverse=True):
        linhas.append(f"  {k}: {r.dist_acertos[k]}")
    linhas.append(f"Proporção de bilhetes com >=11 acertos: {r.proporcao_11_mais:.2%}")
    return "\n".join(linhas)
