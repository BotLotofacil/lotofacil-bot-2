from __future__ import annotations
from dataclasses import dataclass
from typing import List, Set, Tuple, Optional
import numpy as np
import math
import random

UNIFORME = 1.0 / 25.0  # prob. uniforme para cada dezena (1..25)


# =========================
# Configurações
# =========================

@dataclass
class FilterConfig:
    """
    Configuração do filtro pós-geração.
    - paridade_min/max: faixa de nº de pares aceitável na aposta (0..15).
    - col_min/max: faixa por coluna (matriz 5x5; coluna = (d-1)%5) aceitável (0..15).
    - relax_steps: quantos passos de relaxamento aplicar se não houver apostas suficientes.
    """
    paridade_min: int = 6
    paridade_max: int = 9
    col_min: int = 1
    col_max: int = 4
    relax_steps: int = 2


@dataclass
class GeradorApostasConfig:
    """
    Parâmetros do gerador ELITE.
    Ajuste com cautela — esses parâmetros influenciam diretamente a
    força preditiva do algoritmo e o comportamento do MODO ELITE.

    - janela: quantos concursos recentes usar para treinar (>= 50 recomendado).
    - alpha: peso da estimativa vs uniforme (0..0.5 recomendado).
    - min_factor / max_factor: clipping relativo ao uniforme para limitar extremos.
    - repulsao_lift: penalização para pares com lift>1 (evita coocorrências tóxicas).
    - balance_paridade / balance_faixa: penalizações leves para evitar
      composições muito extremas durante a amostragem sequencial.
    - temperatura: controla a aleatoriedade da softmax interna.
    - max_tentativas: tentativas para gerar um bilhete plausível.
    - filtro: regras simples aplicadas após a geração (paridade/colunas).
    - pool_multiplier: tamanho inicial do pool para seleção ELITE.
    - bias_R: peso da repetição, reforçando bilhetes com R próximo de 9–10.
    """

    janela: int = 50
    alpha: float = 0.36
    min_factor: float = 0.60
    max_factor: float = 1.80
    repulsao_lift: float = 0.25
    balance_paridade: float = 0.10
    balance_faixa: float = 0.08
    temperatura: float = 0.90
    max_tentativas: int = 100
    filtro: Optional[FilterConfig] = None

    # pool inicial (modo elite ajusta automaticamente para qtd>=30)
    pool_multiplier: int = 3

    # força do R para o modo elite (9R–10R = alvo central)
    bias_R: float = 0.45


# =========================
# Núcleo do preditor
# =========================

class Predictor:
    """
    Treina a partir de uma janela do histórico (lista de sets de 15 dezenas) e
    gera bilhetes por amostragem sem reposição, aplicando penalizações suaves.

    Agora:
    - continua usando frequências + lift (coocorrência),
    - aplica um viés leve de repetição (R) em relação ao último concurso,
    - e ainda pode usar um filtro pós-geração (paridade/colunas) se configurado.
    """

    def __init__(self, config: GeradorApostasConfig | None = None) -> None:
        self.cfg = config or GeradorApostasConfig()
        self._p: Optional[np.ndarray] = None     # vetor (25,) com probabilidades para cada dezena 1..25
        self._lift: Optional[np.ndarray] = None  # matriz (25,25) com lift de coocorrência
        self._treinado: bool = False
        self._ultimo: Optional[Set[int]] = None  # último resultado considerado na janela

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
        - historico: lista de sets (cada set com 15 dezenas).
        - janela: se None, usa self.cfg.janela.
        """
        if not historico:
            raise ValueError("Histórico vazio.")

        # janela efetiva: valor passado (se houver) ou default da config
        n = int(janela or self.cfg.janela)
        if n < 1:
            n = 1
        if n > len(historico):
            n = len(historico)

        # recorta a janela de interesse
        janelas = historico[-n:] if len(historico) > n else historico

        # guarda o "último resultado" da janela (para viés de repetição R)
        try:
            if janelas:
                # garante estrutura estável: sorted(list(...)) dentro de um set
                self._ultimo = set(sorted(list(janelas[-1])))
            else:
                self._ultimo = None
        except Exception:
            self._ultimo = None

        # estimativas marginais e de coocorrência
        p_raw, lift = self._estimativas_basicas(janelas)

        # Normaliza marginais para distribuição relativa de escolha
        soma = float(p_raw.sum()) + 1e-12
        p_rel = p_raw / soma

        # Mistura com uniforme para reduzir viés exagerado
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

    # ---------- Scoring e amostragem ----------
    def _score_candidato(self, cand: int, selecionados: List[int]) -> float:
        """
        Score do candidato baseado em:
          - log-probabilidade estimada (estável numericamente)
          - leve reforço de repetição (R) em relação ao último concurso
          - penalização por coocorrência excessiva via lift
          - balanceamentos leves de paridade e faixa (1..12 vs 13..25)
        """
        p = float(self._p[cand - 1])
        base = math.log(max(p, 1e-12))

        # Reforço leve para dezenas que se repetem em relação ao último sorteio
        if self._ultimo:
            if cand in self._ultimo:
                # empurra o modelo a colocar mais dezenas repetidas (R alto)
                base += float(self.cfg.bias_R)
            else:
                # leve penalização para completamente "novas"
                base -= float(self.cfg.bias_R) * 0.40

        # Penalização por pares com lift>1 (aparecem juntos acima do esperado)
        rep = 0.0
        if selecionados and self._lift is not None:
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

    # ---------- Filtro pós-geração ----------
    @staticmethod
    def _contar_pares(aposta: List[int]) -> int:
        return sum(1 for d in aposta if d % 2 == 0)

    @staticmethod
    def _contagem_colunas(aposta: List[int]) -> List[int]:
        # colunas 0..4 (Lotofácil 5x5): coluna = (d-1) % 5
        cols = [0, 0, 0, 0, 0]
        for d in aposta:
            cols[(d - 1) % 5] += 1
        return cols

    @classmethod
    def _passa_filtro(cls, aposta: List[int], f: FilterConfig) -> bool:
        pares = cls._contar_pares(aposta)
        if not (f.paridade_min <= pares <= f.paridade_max):
            return False
        cols = cls._contagem_colunas(aposta)
        if any(c < f.col_min or c > f.col_max for c in cols):
            return False
        return True

    @classmethod
    def _aplicar_filtro_pos_geracao(
        cls,
        candidatas: List[List[int]],
        f: FilterConfig,
        qtd_final: int
    ) -> List[List[int]]:
        # 1) tentativa com filtro original
        aprovadas = [a for a in candidatas if cls._passa_filtro(a, f)]
        if len(aprovadas) >= qtd_final:
            return aprovadas[:qtd_final]

        # 2) relaxamentos progressivos
        for step in range(1, f.relax_steps + 1):
            f_relax = FilterConfig(
                paridade_min=max(f.paridade_min - step, 0),
                paridade_max=min(f.paridade_max + step, 15),
                col_min=max(f.col_min - step, 0),
                col_max=min(f.col_max + step, 15),
                relax_steps=f.relax_steps,
            )
            aprovadas = [a for a in candidatas if cls._passa_filtro(a, f_relax)]
            if len(aprovadas) >= qtd_final:
                return aprovadas[:qtd_final]

        # 3) fallback: completa com as originais para garantir qtd_final
        if len(aprovadas) < qtd_final:
            faltantes = qtd_final - len(aprovadas)
            restantes = [a for a in candidatas if a not in aprovadas]
            aprovadas.extend(restantes[:faltantes])
        return aprovadas[:qtd_final]

        # ---------- Geração pública (MODO ELITE) ----------
    def gerar_apostas(self, qtd: int = 5, seed: int | None = None) -> List[List[int]]:
        """
        Gera 'qtd' bilhetes (listas ordenadas de 15 dezenas).

        Estratégia ELITE:
        1) Gera um POOL grande de apostas (pool_multiplier * qtd).
           - se qtd >= 30, usa pool_multiplier “turbo”.
        2) Dá uma NOTA para cada aposta, baseada em:
           - probabilidades p[d] estimadas (força de cada dezena);
           - quantidade de dezenas repetidas em relação ao último concurso (R),
             com alvo em 9R–10R (8 e 11 aceitos como variação).
        3) Ordena da melhor para a pior.
        4) Aplica o filtro pós-geração (paridade/colunas) priorizando as melhores.
        5) Retorna apenas as TOP 'qtd'.

        Objetivo: concentrar força em bilhetes potencialmente explosivos
        (R alto + dezenas fortes), em vez de pulverizar força em bilhetes mornos.
        """
        if not self._treinado:
            raise RuntimeError("Chame fit() antes de gerar.")

        rng = random.Random(seed)

        # 1) Define o tamanho do pool (modo turbo para lotes grandes)
        base_pool = max(1, int(self.cfg.pool_multiplier))
        if qtd >= 30:
            pool_multiplier = max(base_pool * 2, 8)  # ex.: se era 3 => vira 8
        else:
            pool_multiplier = base_pool

        pool = pool_multiplier * max(1, int(qtd))
        candidatas: List[List[int]] = []

        # 2) Geração com checagem leve de plausibilidade (paridade bem ampla)
        for _ in range(pool):
            ok = False
            for _t in range(int(self.cfg.max_tentativas)):
                b = self._amostrar15(rng)
                pares = self._contar_pares(b)
                # filtro leve, amplo, para evitar extremos muito improváveis
                if 2 <= pares <= 13:
                    candidatas.append(b)
                    ok = True
                    break
            if not ok:
                # fallback seguro: uniforme sem reposição
                candidatas.append(sorted(rng.sample(range(1, 26), 15)))

        # 3) Função interna de SCORE para cada aposta do pool (MODO ELITE)
        alvo_R_centro = 9.5   # queremos R ~9–10
        largura_R      = 1.5   # 8..11 ainda são aceitáveis
        bias_R         = float(getattr(self.cfg, "bias_R", 0.45))

        ultimo_set = self._ultimo or set()

        def _score_bilhete(ap: List[int]) -> float:
            # (a) base: força pelas probabilidades estimadas p[d]
            base = 0.0
            if self._p is not None:
                for d in ap:
                    p = float(self._p[d - 1])
                    base += math.log(max(p, 1e-12))

            # (b) reforço por repetição (R) com alvo 9R–10R
            bonus_R = 0.0
            if ultimo_set:
                repetidas = sum(1 for d in ap if d in ultimo_set)

                # janela de interesse em torno de 9.5
                # usamos um "chapéu" (parábola invertida):
                #   score_R = -((R - alvo)^2)  → máximo em R=alvo
                # normalizado por largura_R para não matar 8 e 11
                dist_norm = (repetidas - alvo_R_centro) / largura_R
                score_R_shape = - (dist_norm ** 2)   # máximo ~0 em R≈alvo; cai pros extremos

                # camada extra: penaliza R muito baixos (<=6) e muito altos (>=13)
                if repetidas <= 6 or repetidas >= 13:
                    score_R_shape -= 2.0

                bonus_R = bias_R * score_R_shape

            return base + bonus_R

        # 4) Ordena o pool pela nota (da melhor para a pior)
        candidatas.sort(key=_score_bilhete, reverse=True)

        # 5) Aplica filtro pós-geração, priorizando as melhores
        if self.cfg.filtro is not None:
            apostas = self._aplicar_filtro_pos_geracao(
                candidatas,
                self.cfg.filtro,
                int(qtd)
            )
        else:
            apostas = candidatas[: int(qtd)]

        return apostas

        except Exception:
            logger.error("Erro ao gerar apostas:\n" + traceback.format_exc())
            await update.message.reply_text("Erro ao gerar apostas. Tente novamente.")
