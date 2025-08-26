# utils/predictor.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Set, Tuple
import numpy as np
import math
import random

UNIFORME = 1.0 / 25.0  # prob. uniforme para cada dezena (1..25)

@dataclass
class GeradorApostasConfig:
    """
    Parâmetros do gerador. Ajuste com cautela.
    - janela: quantos concursos recentes usar para treinar (>= 50 recomendado).
    - alpha: peso da estimativa vs uniforme (0..0.5 recomendado).
    - min_factor / max_factor: clipping relativo ao uniforme para limitar extremos.
    - repulsao_lift: força de penalização de pares com lift>1 (aparecem juntos além do esperado).
    - balance_paridade / balance_faixa: penalizações leves para não desequilibrar composição.
    - temperatura: suaviza/acentua diferenças de score na escolha sequencial.
    - max_tentativas: robustez na geração de cada bilhete.
    """
    janela: int = 100
    alpha: float = 0.45
    min_factor: float = 0.60
    max_factor: float = 1.80
    repulsao_lift: float = 0.25
    balance_paridade: float = 0.10
    balance_faixa: float = 0.08
    temperatura: float = 0.90
    max_tentativas: int = 100

class Predictor:
    """
    Treina a partir de uma janela do histórico (lista de sets de 15 dezenas) e
    gera bilhetes por amostragem sem reposição, aplicando penalizações suaves.
    """
    def __init__(self, config: GeradorApostasConfig | None = None) -> None:
        self.cfg = config or GeradorApostasConfig()
        self._p = None            # vetor (25,) com probabilidades para cada dezena 1..25
        self._lift = None         # matriz (25,25) com lift de coocorrência
        self._treinado = False

    # ---------- Treino ----------
    @staticmethod
    def _estimativas_basicas(janelas: List[Set[int]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula:
          - p_raw: frequência marginal de cada dezena (aprox prob de estar entre as 15 do sorteio)
          - lift: matriz de coocorrência normalizada por marginais (Pij / (Pi*Pj))
        """
        n = len(janelas)
        if n == 0:
            raise ValueError("Janela vazia para estimativa.")

        cnt = np.zeros(26, dtype=float)   # índices 1..25
        pair = np.zeros((26, 26), dtype=float)

        for s in janelas:
            arr = sorted(list(s))
            for x in arr:
                cnt[x] += 1.0
                pair[x, x] += 1.0
            for i in range(len(arr)):
                xi = arr[i]
                for j in range(i + 1, len(arr)):
                    xj = arr[j]
                    pair[xi, xj] += 1.0
                    pair[xj, xi] += 1.0

        p_raw = cnt[1:] / n              # (25,)
        pij = pair[1:, 1:] / n           # (25,25)
        eps = 1e-9
        denom = (p_raw[:, None] * p_raw[None, :]) + eps
        lift = pij / denom               # (25,25)
        return p_raw, lift

    def fit(self, historico: List[Set[int]], janela: int | None = None) -> None:
        """
        Prepara _p (probabilidades misturadas e clipadas) e _lift.
        """
        if not historico:
            raise ValueError("Histórico vazio.")

        # janela efetiva: valor passado (se houver) ou default da config
        n = int(janela or self.cfg.janela)
        if n < 1:
            n = 1
        if n > len(historico):
            n = len(historico)

        janelas = historico[-n:] if len(historico) > n else historico

        p_raw, lift = self._estimativas_basicas(janelas)

        # Normaliza marginais para distribuição relativa de escolha
        soma = float(p_raw.sum()) + 1e-12
        p_rel = p_raw / soma

        # Mistura com uniforme para reduzir viés
        alpha = max(0.0, min(float(self.cfg.alpha), 0.5))
        p_mix = (1.0 - alpha) * (np.ones(25) * UNIFORME) + alpha * p_rel

        # Clipping relativo ao uniforme
        p_min = float(self.cfg.min_factor) * UNIFORME
        p_max = float(self.cfg.max_factor) * UNIFORME
        p_mix = np.clip(p_mix, p_min, p_max)
        p_mix = p_mix / (p_mix.sum() + 1e-12)

        self._p = p_mix.astype(float)
        self._lift = lift.astype(float)
        self._treinado = True

    # ---------- Geração ----------
    def _score_candidato(self, cand: int, selecionados: List[int]) -> float:
        """
        Score do candidato baseado em:
          - log-probabilidade estimada (estável numericamente)
          - penalização por coocorrência excessiva via lift
          - balanceamentos leves de paridade e faixa (1..12 vs 13..25)
        """
        p = float(self._p[cand - 1])
        base = math.log(max(p, 1e-12))

        # Penalização por pares com lift>1 (aparecem juntos acima do esperado)
        rep = 0.0
        if selecionados:
            for j in selecionados:
                lij = float(self._lift[cand - 1, j - 1])
                if lij > 1.0:
                    rep -= float(self.cfg.repulsao_lift) * math.log(lij + 1e-12)

        # Balanceamento leve de paridade (evitar extremos)
        impares_sel = sum(1 for x in selecionados if x % 2 == 1)
        pares_sel = len(selecionados) - impares_sel
        impar = (cand % 2 == 1)
        pen_paridade = 0.0
        if impar and impares_sel > pares_sel + 1:
            pen_paridade -= float(self.cfg.balance_paridade)
        if (not impar) and pares_sel > impares_sel + 1:
            pen_paridade -= float(self.cfg.balance_paridade)

        # Balanceamento leve de faixa (1..12 vs 13..25)
        abaixo = sum(1 for x in selecionados if x <= 12)
        acima = len(selecionados) - abaixo
        abaixo_12 = (cand <= 12)
        pen_faixa = 0.0
        if abaixo_12 and abaixo > acima + 1:
            pen_faixa -= float(self.cfg.balance_faixa)
        if (not abaixo_12) and acima > abaixo + 1:
            pen_faixa -= float(self.cfg.balance_faixa)

        return base + rep + pen_paridade + pen_faixa

    def _amostrar15(self, rng: random.Random) -> List[int]:
        """
        Geração sequencial sem reposição:
          - a cada passo, calcula score para os candidatos remanescentes;
          - escolhe via softmax controlado por 'temperatura';
          - repete até 15 dezenas.
        """
        disponiveis = list(range(1, 26))
        selecionados: List[int] = []
        temp = max(0.3, float(self.cfg.temperatura))

        for _ in range(15):
            scores = [self._score_candidato(c, selecionados) for c in disponiveis]
            m = max(scores)
            exps = [math.exp((s - m) / temp) for s in scores]
            soma = sum(exps) + 1e-12
            probs = [e / soma for e in exps]
            cand = rng.choices(disponiveis, weights=probs, k=1)[0]
            selecionados.append(cand)
            disponiveis.remove(cand)

        selecionados.sort()
        return selecionados

    def gerar_apostas(self, qtd: int = 5, seed: int | None = None) -> List[List[int]]:
        """
        Gera 'qtd' bilhetes (listas ordenadas de 15 dezenas).
        Aplica checagens leves para evitar extremos improváveis.
        Se falhar todas as tentativas, faz fallback uniforme.
        """
        if not self._treinado:
            raise RuntimeError("Chame fit() antes de gerar.")

        rng = random.Random(seed)
        bilhetes: List[List[int]] = []
        for _ in range(max(1, int(qtd))):
            ok = False
            for _t in range(int(self.cfg.max_tentativas)):
                b = self._amostrar15(rng)
                pares = sum(1 for x in b if x % 2 == 0)
                # filtro leve de plausibilidade; não engessa
                if 3 <= pares <= 12:
                    bilhetes.append(b)
                    ok = True
                    break
            if not ok:
                # fallback seguro: uniforme sem reposição
                bilhetes.append(sorted(rng.sample(range(1, 26), 15)))
        return bilhetes
