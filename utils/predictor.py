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
    Parâmetros do gerador. Ajuste com cautela.
    - janela: quantos concursos recentes usar para treinar (>= 50 recomendado).
    - alpha: peso da estimativa vs uniforme (0..0.5 recomendado).
    - min_factor / max_factor: clipping relativo ao uniforme para limitar extremos.
    - repulsao_lift: penalização de pares com lift>1 (aparecem juntos além do esperado).
    - balance_paridade / balance_faixa: penalizações leves para não desequilibrar composição.
    - temperatura: suaviza/acentua diferenças de score na escolha sequencial.
    - max_tentativas: robustez na geração de cada bilhete.
    - filtro: regras simples de qualidade aplicadas após a geração.
    - pool_multiplier: fator para gerar um pool maior e então filtrar (>=1).
    - bias_R: reforço leve para dezenas do último concurso (repetição).
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
    pool_multiplier: int = 3
    bias_R: float = 0.35  # reforço para dezenas que se repetem do último concurso


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

        janelas = historico[-n:] if len(historico) > n else historico

        # guarda o "último resultado" da janela (para viés de repetição)
        try:
            self._ultimo = set(janelas[-1]) if janelas else None
        except Exception:
            self._ultimo = None

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

    # --- /gerar: rápido, estável, sem cache e com diversidade entre chamadas ---
    async def gerar_apostas(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Comando /gerar – Estratégia Mestre, rápido e estável.

        • Usa o NOVO PREDITOR ELITE (utils.predictor.Predictor):
          - gera um pool grande;
          - pontua cada bilhete (probabilidade + repetição R);
          - escolhe apenas as TOP apostas do pool.
        • α TRAVADO = 0.36 (LOCK_ALPHA_GERAR=True), independente do aprendizado.
        • Paridade alvo FINAL: 7–8 | Máx. sequência FINAL: ≤3 | Anti-overlap FINAL: ≤11.
        • Repetição R: foco em 9R–10R, com 1×8R e 1×11R de variação (garantido nos templates).
        • Uso: /gerar [qtd] [janela] [alpha]  → (alpha é ignorado: lock=0.36)
        • Padrão: 5 apostas | janela=60 | α=0.36 (travado no /gerar)
        """

        import asyncio, traceback
        from datetime import datetime
        from zoneinfo import ZoneInfo

        user_id = update.effective_user.id
        if not self._usuario_autorizado(user_id):
            return await update.message.reply_text("⛔ Você não está autorizado a gerar apostas.")

        # >>> anti-abuso
        if not self._is_admin(user_id):
            if _is_temporarily_blocked(user_id):
                return await update.message.reply_text("🚫 Você está temporariamente bloqueado por excesso de tentativas.")
            allowed, warn = _register_command_event(user_id, is_unknown=False)
            if not allowed:
                return await update.message.reply_text(warn)
            if warn:
                await update.message.reply_text(warn)
        # <<< anti-abuso

        # Defaults
        qtd, janela, alpha = QTD_BILHETES_PADRAO, JANELA_PADRAO, ALPHA_PADRAO

        # Parse argumentos posicionais (opcionais)
        try:
            if context.args and len(context.args) >= 1:
                qtd = int(context.args[0])
            if context.args and len(context.args) >= 2:
                janela = int(context.args[1])
            if context.args and len(context.args) >= 3:
                alpha = float(context.args[2].replace(",", "."))
        except Exception:
            # se der erro, mantém defaults
            pass

        # Clamps defensivos
        qtd, janela, alpha = self._clamp_params(qtd, janela, alpha)
        target_qtd = max(1, int(qtd))  # garante respeitar /gerar 50, etc.

        # >>> trava α somente no /gerar (sem afetar demais comandos)
        alpha = self._alpha_para_comando("/gerar", alpha_sugerido=alpha)
        # <<< trava α somente no /gerar

        # --- coerência de estado e alpha_usado ---
        st = _normalize_state_defaults(_bolao_load_state() or {})
        st = self._coagir_estado_lock_alpha(st)
        alpha_usado = self._alpha_para_execucao(st)
        try:
            _bolao_save_state(st)
        except Exception:
            pass

        # Histórico/último seguro
        try:
            historico = carregar_historico(HISTORY_PATH)
        except Exception:
            historico = []
        try:
            ultimo = self._ultimo_resultado(historico) if historico else []
        except Exception:
            ultimo = []

        u_set = set(ultimo)
        universo = list(range(1, 26))

        # --------- utilidades canônicas e selagem ----------
        def _canon(a: list[int]) -> list[int]:
            """Normaliza: 1..25, únicos, ordenados, exatamente 15."""
            a = [int(x) for x in a if 1 <= int(x) <= 25]
            a = sorted(set(a))
            if len(a) > 15:
                keep = []
                for n in a:
                    if len(keep) == 15:
                        break
                    # prioriza alternância entre números do último resultado e fora dele
                    if (len(keep) % 2 == 0 and n in u_set) or (len(keep) % 2 == 1 and n not in u_set):
                        keep.append(n)
                if len(keep) < 15:
                    for n in a:
                        if n not in keep:
                            keep.append(n)
                            if len(keep) == 15:
                                break
                a = keep
            elif len(a) < 15:
                comp = [n for n in universo if n not in a]
                # tenta completar sem criar sequências longas
                for n in comp:
                    if (n - 1 not in a) and (n + 1 not in a):
                        a.append(n)
                        if len(a) == 15:
                            break
                if len(a) < 15:
                    for n in comp:
                        if n not in a:
                            a.append(n)
                            if len(a) == 15:
                                break
                a = sorted(a)
            return a

        def _selar(a: list[int]) -> list[int]:
            """
            Canônico + lock forte (pares 7–8, seq≤3), preservando ao máximo
            a inteligência vinda do preditor (não reembaralha à toa).
            """
            a = _canon(a)
            try:
                a = self._hard_lock_fast(a, ultimo, anchors=frozenset())
            except Exception:
                a = self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3)
            return _canon(a)

        # --------- “sal” por chamada para variar offsets do fallback ----------
        try:
            snap = self._latest_snapshot()
            snap_id = getattr(snap, "snapshot_id", "n/a")
        except Exception:
            snap_id = "n/a"
        call_salt = self._next_draw_seed(str(snap_id))  # contador persistido por snapshot

        # --------- Fallback determinístico (rápido), mas salgado por chamada ----------
        def _fallback(qty: int, salt: int) -> list[list[int]]:
            base = []
            L = list(ultimo) or universo[:15]
            C = [n for n in universo if n not in L]
            for i in range(max(1, qty)):
                offL = (salt + i * 3) % len(L)
                offC = (salt // 7 + i * 5) % len(C) if C else 0
                a = (L[offL:] + L[:offL])[:8] + (C[offC:] + C[:offC])[:7]
                base.append(_selar(a))
            return base

        # --------- Preditor SEM cache (sempre gera lote novo) ----------
        async def _run_preditor():
            """
            Chama o NOVO núcleo preditivo (_gerar_apostas_inteligentes),
            que por sua vez usa o Predictor ELITE (pool + score + TOP apostas).
            """
            return await asyncio.to_thread(self._gerar_apostas_inteligentes, target_qtd, janela, alpha)

        # --------- Pipeline principal ----------
        try:
            # 0) Preditor com timeout + fallback determinístico
            try:
                brutas = await asyncio.wait_for(_run_preditor(), timeout=2.5)
            except asyncio.TimeoutError:
                logger.warning("Predictor >2.5s: usando fallback determinístico.")
                brutas = _fallback(target_qtd, call_salt)
            except Exception:
                logger.warning("Predictor falhou: usando fallback determinístico.", exc_info=True)
                brutas = _fallback(target_qtd, call_salt)

            # 1) Selagem por aposta (rápida)
            apostas = [_selar(a) for a in brutas]

            # 2) Reposição até atingir 'target_qtd' (variações determinísticas)
            rep_salt = call_salt
            seen = {tuple(x) for x in apostas}
            while len(apostas) < target_qtd:
                rep_salt += 1
                extra = _fallback(1, rep_salt)[0]
                t = tuple(extra)
                if t not in seen:
                    apostas.append(_selar(extra))
                    seen.add(t)

            # 3) Pós-filtro unificado (forma + dedup/overlap + bias + forma)
            if ultimo:
                try:
                    apostas = self._pos_filtro_unificado(apostas, ultimo)
                except Exception:
                    logger.warning("pos_filtro_unificado falhou; aplicando hard_lock por aposta.", exc_info=True)
                    apostas = [self._hard_lock_fast(a, ultimo, anchors=frozenset()) for a in apostas]
            else:
                # histórico indisponível: aplica ao menos o hard_lock
                apostas = [self._hard_lock_fast(a, ultimo=[], anchors=frozenset()) for a in apostas]

            # [NOVO] Pós-filtro determinístico (anti-overlap>11 e seq>3)
            try:
                apostas = self._pos_filtro_unificado_deterministico(apostas)
            except Exception:
                logger.warning("pos_filtro_unificado_deterministico falhou; seguindo sem ajuste adicional.", exc_info=True)

            # 3.0b) Força anti-overlap ≤ limite (sem perder shape Mestre)
            try:
                limite_overlap_inicial = int(globals().get("BOLAO_MAX_OVERLAP", 11))
            except Exception:
                limite_overlap_inicial = 11
            try:
                apostas = self._forcar_anti_overlap(apostas, ultimo=ultimo or [], limite=limite_overlap_inicial)
            except Exception:
                logger.warning("forcar_anti_overlap falhou; seguindo sem ajuste adicional.", exc_info=True)

            # =====================================================================
            # --------------------  SELAGEM DE SAÍDA (NOVO)  ----------------------
            # Garante: paridade 7–8, seq≤3 e anti-overlap≤11 ANTES de persistir/mostrar
            try:
                OVERLAP_MAX = int(globals().get("BOLAO_MAX_OVERLAP", 11))
            except Exception:
                OVERLAP_MAX = 11

            def _shape_ok(a: list[int]) -> bool:
                return self._shape_ok_basico(a)

            # 1) Funil Mestre (se falhar, cai no fallback básico)
            try:
                apostas_ok = self._finalizar_lote_mestre(
                    apostas=apostas,
                    ultimo=ultimo or [],
                    target_qtd=target_qtd,
                    call_salt=call_salt,
                    overlap_max=OVERLAP_MAX,
                    max_ciclos=8,
                    aplicar_cap_par=True,
                )
            except Exception:
                logger.warning("_finalizar_lote_mestre falhou; aplicando fallback básico.", exc_info=True)
                apostas_ok = [self._hard_lock_fast(a, ultimo=ultimo or [], anchors=frozenset()) for a in apostas]
                try:
                    apostas_ok = self._forcar_anti_overlap(apostas_ok, ultimo=ultimo or [], limite=OVERLAP_MAX)
                except Exception:
                    pass

            # 2) FECHAMENTO STRICTO: força passar no TRIPLO CHECK (ou aproxima)
            apostas_ok = self._fechar_lote_stricto(
                apostas_ok,
                ultimo=ultimo or [],
                overlap_max=OVERLAP_MAX,
                max_ciclos=8
            )

            # 3) Garantia de quantidade exata (se necessário) + fechamento final curto
            if len(apostas_ok) < target_qtd:
                rep_salt = call_salt
                seen = {tuple(sorted(a)) for a in apostas_ok}
                while len(apostas_ok) < target_qtd:
                    rep_salt += 1
                    cand = _fallback(1, rep_salt)[0]
                    try:
                        cand = self._hard_lock_fast(cand, ultimo=ultimo or [], anchors=frozenset())
                    except Exception:
                        cand = self._ajustar_paridade_e_seq(cand, alvo_par=(7, 8), max_seq=3)
                    cand = sorted(set(cand))
                    if not _shape_ok(cand):
                        continue
                    if all(len(set(cand) & set(b)) <= OVERLAP_MAX for b in apostas_ok):
                        t = tuple(cand)
                        if t not in seen:
                            apostas_ok.append(cand)
                            seen.add(t)

            # 4) Selagem final + dedup + anti-overlap final (idempotente)
            try:
                apostas_ok = [self._hard_lock_fast(a, ultimo=ultimo or [], anchors=frozenset()) for a in apostas_ok]
            except Exception:
                apostas_ok = [self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3) for a in apostas_ok]

            uniq, seen = [], set()
            for a in apostas_ok:
                t = tuple(a)
                if t not in seen:
                    seen.add(t)
                    uniq.append(a)
            apostas_ok = uniq

            try:
                apostas_ok = self._forcar_anti_overlap(apostas_ok, ultimo=ultimo or [], limite=OVERLAP_MAX)
            except Exception:
                pass

            # valida forma novamente
            apostas_ok = [
                (self._hard_lock_fast(a, ultimo=ultimo or [], anchors=frozenset())
                 if not _shape_ok(a) else a)
                for a in apostas_ok
            ]

            # Usa a versão selada e reparada
            apostas = [sorted(a) for a in apostas_ok]
            # ------------------  FIM SELAGEM DE SAÍDA (NOVO)  --------------------
            # =====================================================================

            # --- persistência para o auto_aprender: last_generation ---
            try:
                st2 = _normalize_state_defaults(_bolao_load_state() or {})
                st2.setdefault("learning", {})["last_generation"] = {
                    "apostas": apostas_ok  # lista de 15 números (ordenados) por aposta
                }
                _bolao_save_state(st2)
            except Exception:
                logger.warning("Falha ao persistir learning.last_generation.", exc_info=True)

            # 3.2) REGISTRO para aprendizado leve
            try:
                self._registrar_geracao(apostas_ok, base_resultado=ultimo or [])
            except Exception:
                logger.warning("Falha ao registrar geração para aprendizado leve (/gerar).", exc_info=True)

            # >>> registrar o lote no estado (pending_batches)
            try:
                st3 = _normalize_state_defaults(_bolao_load_state() or {})
                batches = st3.get("pending_batches", [])
                batches.append({
                    "ts": datetime.now(ZoneInfo(TIMEZONE)).isoformat(),
                    "snapshot": getattr(self._latest_snapshot(), "snapshot_id", "--"),
                    "alpha": float(st3.get("alpha", ALPHA_PADRAO)),
                    "janela": int((st3.get("learning") or {}).get("janela", JANELA_PADRAO)),
                    "oficial_base": " ".join(f"{n:02d}" for n in (ultimo or [])),
                    "qtd": len(apostas_ok),
                    "apostas": [" ".join(f"{x:02d}" for x in a) for a in apostas_ok],
                })
                st3["pending_batches"] = batches[-100:]
                _bolao_save_state(st3)
            except Exception:
                logger.warning("Falha ao registrar pending_batch.", exc_info=True)

            # --- Mensagem "Aprendizado leve atualizado" com média REAL do lote persistido ---
            try:
                media_real = self._media_real_do_lote_persistido()

                st_msg = _normalize_state_defaults(_bolao_load_state() or {})
                st_msg = self._coagir_estado_lock_alpha(st_msg)
                learn_msg = (st_msg.get("learning") or {})
                bias_meta = learn_msg.get("bias_meta", {}) or {}
                alpha_usado_msg = float(st_msg["runtime"].get("alpha_usado", ALPHA_LOCK_VALUE))
                alpha_proposto = learn_msg.get("alpha_proposto", None)
                lock_ativo = st_msg["locks"].get("alpha_travado", True)

                if lock_ativo:
                    alpha_info = f"α usado: {alpha_usado_msg:.2f} (travado)"
                    if alpha_proposto is not None:
                        alpha_info += f" | α proposto: {float(alpha_proposto):.2f} (pendente)"
                else:
                    alpha_info = f"α usado: {alpha_usado_msg:.2f} (livre)"

                msg = (
                    "📈 Aprendizado leve atualizado.\n"
                    f"• Lote avaliado: {len(apostas_ok)} apostas\n"
                    f"• Média de acertos: {media_real:.2f}\n"
                    f"• {alpha_info}\n"
                    f"• bias[R]={bias_meta.get('R', 0.0):+.3f}  "
                    f"bias[par]={bias_meta.get('par', 0.0):+.3f}  "
                    f"bias[seq]={bias_meta.get('seq', 0.0):+.3f}"
                )
                await update.message.reply_text(msg)
            except Exception:
                logger.warning("Falha ao compor/enviar mensagem de aprendizado leve.", exc_info=True)

            # 4) Formatação + envio (usa α efetivo do /gerar)
            try:
                resposta = self._formatar_resposta(apostas_ok, janela, alpha_usado)
            except Exception:
                # Fallback de formatação (mantém seu visual atual)
                linhas = ["🎰 <b>SUAS APOSTAS INTELIGENTES</b> 🎰\n"]
                for i, a in enumerate(apostas_ok, 1):
                    pares = self._contar_pares(a) if hasattr(self, "_contar_pares") else sum(1 for n in a if n % 2 == 0)
                    seq = self._max_seq(a) if hasattr(self, "_max_seq") else 0
                    linhas.append(
                        f"<b>Aposta {i}:</b> {' '.join(f'{n:02d}' for n in a)}\n"
                        f"🔢 Pares: {pares} | Ímpares: {15 - pares} | SeqMax: {seq}\n"
                    )
                if SHOW_TIMESTAMP:
                    now_sp = datetime.now(ZoneInfo(TIMEZONE))
                    carimbo = now_sp.strftime("%Y-%m-%d %H:%M:%S %Z")
                    linhas.append(f"<i>janela={janela} | α={alpha_usado:.2f}</i>")
                resposta = "\n".join(linhas)

            # 5) Saída
            await self._send_long(update, resposta, parse_mode="HTML")

            # Opcional: auto_aprender (com gating ativo ele retorna sem mexer)
            try:
                await self.auto_aprender(update, context)
            except Exception:
                logger.warning("auto_aprender falhou; prosseguindo normalmente.", exc_info=True)

        except Exception:
            logger.error("Erro ao gerar apostas:\n" + traceback.format_exc())
            await update.message.reply_text("Erro ao gerar apostas. Tente novamente.")
