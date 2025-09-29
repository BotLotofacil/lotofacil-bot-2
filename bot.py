# bot.py

import os
import logging
import traceback
import asyncio
import re
import hashlib
from functools import partial
from typing import List, Set, Tuple
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from os import getenv

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from dotenv import load_dotenv

# ========================
# Carrega variáveis de ambiente locais
# ========================
load_dotenv()

# ========================
# Logging
# ========================
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ========================
# Imports do projeto (compatíveis: utils.* OU raiz)
# ========================
try:
    # layout em pacote
    from utils.history import carregar_historico, ultimos_n_concursos
    from utils.predictor import Predictor, GeradorApostasConfig, FilterConfig
    from utils.backtest import executar_backtest_resumido
    LAYOUT = "utils"
except Exception:
    # layout na raiz
    from history import carregar_historico, ultimos_n_concursos
    from predictor import Predictor, GeradorApostasConfig, FilterConfig
    from backtest import executar_backtest_resumido
    LAYOUT = "root"
    logger.info("Usando layout de módulos na raiz (history.py/predictor.py/backtest.py).")

# ========================
# Parâmetros padrão do gerador
# ========================
JANELA_PADRAO = 50
ALPHA_PADRAO = 0.55
QTD_BILHETES_PADRAO = 5

SHOW_TIMESTAMP = True
TIMEZONE = "America/Sao_Paulo"

# Limites defensivos
JANELA_MIN, JANELA_MAX = 50, 1000
ALPHA_MIN, ALPHA_MAX = 0.05, 0.95
BILH_MIN, BILH_MAX   = 1, 20

HISTORY_PATH = "data/history.csv"
WHITELIST_PATH = "whitelist.txt"

# Cooldown (segundos) para evitar flood
COOLDOWN_SECONDS = 10

# Identificação do build (para /versao)
BUILD_TAG = getenv("BUILD_TAG", "unknown")

# ========================
# Bot Principal
# ========================
class LotoFacilBot:
    def __init__(self):
        self.token = self._get_bot_token()
        self.admin_id = self._get_admin_id()
        self.whitelist_path = WHITELIST_PATH

        # Garante pastas/arquivos essenciais antes de qualquer IO
        self._ensure_paths()

        self.whitelist = self._carregar_whitelist()
        self._garantir_admin_na_whitelist()

        self.app = ApplicationBuilder().token(self.token).build()
        # mapa de cooldown: {(chat_id, comando): timestamp}
        self._cooldown_map = {}
        self._setup_handlers()

    # ------------- Utilidades internas -------------
    def _get_bot_token(self) -> str:
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            raise EnvironmentError("Variável TELEGRAM_BOT_TOKEN não configurada.")
        return token

    def _get_admin_id(self) -> int:
        admin_id = os.getenv("ADMIN_TELEGRAM_ID")
        if not admin_id or not admin_id.isdigit():
            raise EnvironmentError("ADMIN_TELEGRAM_ID não configurado corretamente.")
        return int(admin_id)

    def _ensure_paths(self):
        """Garante que diretórios e arquivos base existam (sem criar history.csv)."""
        # pasta data/
        data_dir = Path(HISTORY_PATH).parent
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.warning(f"Não foi possível criar pasta de dados: {data_dir}")

        # pasta do whitelist.txt (caso WHITELIST_PATH esteja em subpasta)
        wl_dir = Path(self.whitelist_path).parent
        try:
            wl_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.warning(f"Não foi possível criar pasta do whitelist: {wl_dir}")

        # se whitelist.txt não existir, cria vazio (admin será adicionado depois)
        wl_file = Path(self.whitelist_path)
        if not wl_file.exists():
            try:
                wl_file.write_text("", encoding="utf-8")
            except Exception:
                logger.warning("Não foi possível criar arquivo whitelist.txt")

    def _carregar_whitelist(self) -> Set[int]:
        """Carrega os IDs autorizados do arquivo de whitelist."""
        if not os.path.exists(self.whitelist_path):
            return set()
        with open(self.whitelist_path, "r", encoding="utf-8") as f:
            return set(int(l.strip()) for l in f if l.strip().isdigit())

    def _salvar_whitelist(self):
        """Salva a whitelist atual no arquivo (gravação atômica)."""
        tmp_path = self.whitelist_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for user_id in sorted(self.whitelist):
                f.write(f"{user_id}\n")
        os.replace(tmp_path, self.whitelist_path)

    def _garantir_admin_na_whitelist(self):
        """Garante que o administrador esteja sempre autorizado."""
        if self.admin_id not in self.whitelist:
            self.whitelist.add(self.admin_id)
            self._salvar_whitelist()
            logger.info(f"Administrador {self.admin_id} autorizado automaticamente.")

    def _usuario_autorizado(self, user_id: int) -> bool:
        return user_id in self.whitelist

    def _is_admin(self, user_id: int) -> bool:
        return user_id == self.admin_id

    def _hit_cooldown(self, chat_id: int, comando: str) -> bool:
        """
        Retorna True se o cooldown ainda estiver ativo para (chat_id, comando).
        Caso contrário, atualiza o timestamp e retorna False.
        """
        import time
        key = (chat_id, comando)
        now = time.time()
        last = self._cooldown_map.get(key, 0)
        if now - last < COOLDOWN_SECONDS:
            return True
        self._cooldown_map[key] = now
        return False

    # --------- Validações e clamps de parâmetros ---------
    def _clamp_params(self, qtd: int, janela: int, alpha: float) -> Tuple[int, int, float]:
        qtd = max(BILH_MIN, min(BILH_MAX, int(qtd)))
        janela = max(JANELA_MIN, min(JANELA_MAX, int(janela)))
        alpha = float(alpha)
        if alpha < ALPHA_MIN or alpha > ALPHA_MAX:
            alpha = ALPHA_PADRAO
        return qtd, janela, alpha

    # ------------- Gerador preditivo -------------
    def _gerar_apostas_inteligentes(
        self,
        qtd: int = QTD_BILHETES_PADRAO,
        janela: int = JANELA_PADRAO,
        alpha: float = ALPHA_PADRAO,
    ) -> List[List[int]]:
        """
        Gera bilhetes usando o preditor configurado.
        - Aplica pool (3x) e filtro pós-geração (pares 6–9; colunas 1–4; relaxamento).
        Em caso de falha, aplica fallback uniforme.
        """
        try:
            qtd, janela, alpha = self._clamp_params(qtd, janela, alpha)
            historico = carregar_historico(HISTORY_PATH)
            janela_hist = ultimos_n_concursos(historico, janela)

            filtro = FilterConfig(
                paridade_min=6,
                paridade_max=9,
                col_min=1,
                col_max=4,
                relax_steps=2,
            )

            cfg = GeradorApostasConfig(
                janela=janela,
                alpha=alpha,
                filtro=filtro,
                pool_multiplier=3,
            )
            modelo = Predictor(cfg)
            modelo.fit(janela_hist, janela=janela)
            return modelo.gerar_apostas(qtd=qtd)
        except Exception:
            logger.error("Falha no gerador preditivo; aplicando fallback.\n" + traceback.format_exc())
            import random
            rng = random.Random()
            return [sorted(rng.sample(range(1, 26), 15)) for _ in range(max(1, qtd))]

    # ------------- Parse utilitário p/ backtest -------------
    def _parse_backtest_args(self, args: List[str]) -> Tuple[int, int, float]:
        """
        Aceita:
          - Posicional: /backtest [janela] [bilhetes_por_concurso] [alpha]
          - Chave=valor: /backtest janela=200 bilhetes=5 alpha=0,30
          - Aliases: j=, b=, a=
        Retorna parâmetros validados.
        """
        janela = JANELA_PADRAO
        bilhetes_por_concurso = QTD_BILHETES_PADRAO
        alpha = ALPHA_PADRAO

        if not args:
            return janela, bilhetes_por_concurso, alpha

        joined = " ".join(args).strip().replace(",", ".")
        joined = re.sub(r"\bj\s*=", "janela=", joined)
        joined = re.sub(r"\bb(ilhetes)?\s*=", "bilhetes=", joined)
        joined = re.sub(r"\ba\s*=", "alpha=", joined)
        has_kv = bool(re.search(r"\b(janela|bilhetes|alpha)\s*=", joined))

        if has_kv:
            m_j = re.search(r"\bjanela\s*=\s*(\d{1,5})\b", joined)
            if m_j: janela = int(m_j.group(1))
            m_b = re.search(r"\bbilhetes\s*=\s*(\d{1,3})\b", joined)
            if m_b: bilhetes_por_concurso = int(m_b.group(1))
            m_a = re.search(r"\balpha\s*=\s*([01]?(?:\.\d+)?)\b", joined)
            if m_a: alpha = float(m_a.group(1))
        else:
            try:
                if len(args) >= 1: janela = int(args[0])
                if len(args) >= 2: bilhetes_por_concurso = int(args[1])
                if len(args) >= 3: alpha = float(args[2].replace(",", "."))
            except Exception:
                pass

        # Clamp final
        bilhetes_por_concurso, janela, alpha = self._clamp_params(bilhetes_por_concurso, janela, alpha)
        return janela, bilhetes_por_concurso, alpha

    # ------------- Handlers -------------
    def _setup_handlers(self):
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("gerar", self.gerar_apostas))
        self.app.add_handler(CommandHandler("meuid", self.meuid))
        self.app.add_handler(CommandHandler("autorizar", self.autorizar))
        self.app.add_handler(CommandHandler("remover", self.remover))
        self.app.add_handler(CommandHandler("backtest", self.backtest))  # oculto (só admin)
        # --- Novo handler: /mestre ---
        self.app.add_handler(CommandHandler("mestre", self.mestre))
        # Diagnóstico
        self.app.add_handler(CommandHandler("ping", self.ping))
        self.app.add_handler(CommandHandler("versao", self.versao))
        logger.info("Handlers ativos: /start /gerar /mestre /meuid /autorizar /remover /backtest /ping /versao")

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /start – mensagem de boas-vindas e aviso legal."""
        mensagem = (
            "⚠️ <b>Aviso Legal</b>\n"
            "Este bot é apenas para fins estatísticos e recreativos. "
            "Não há garantia de ganhos na Lotofácil.\n\n"
            "🎉 <b>Bem-vindo</b>\n"
            "Use /gerar para receber 5 apostas baseadas em 100 concursos e α=0,30.\n"
            "Use /meuid para obter seu identificador e solicitar autorização.\n"
        )
        await update.message.reply_text(mensagem, parse_mode="HTML")

    async def gerar_apostas(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Comando /gerar – Gera apostas inteligentes.
        Uso: /gerar [qtd] [janela] [alpha]
        Padrão: 5 apostas | janela=50 | α=0,55
        """
        user_id = update.effective_user.id
        if not self._usuario_autorizado(user_id):
            await update.message.reply_text("⛔ Você não está autorizado a gerar apostas.")
            return

        qtd, janela, alpha = QTD_BILHETES_PADRAO, JANELA_PADRAO, ALPHA_PADRAO

        try:
            if context.args and len(context.args) >= 1: qtd = int(context.args[0])
            if context.args and len(context.args) >= 2: janela = int(context.args[1])
            if context.args and len(context.args) >= 3: alpha = float(context.args[2].replace(",", "."))
        except Exception:
            pass

        qtd, janela, alpha = self._clamp_params(qtd, janela, alpha)

        try:
            apostas = self._gerar_apostas_inteligentes(qtd=qtd, janela=janela, alpha=alpha)
            resposta = self._formatar_resposta(apostas, janela, alpha)
            await update.message.reply_text(resposta, parse_mode="HTML")
        except Exception:
            logger.error("Erro ao gerar apostas:\n" + traceback.format_exc())
            await update.message.reply_text("Erro ao gerar apostas. Tente novamente.")

    def _formatar_resposta(self, apostas: List[List[int]], janela: int, alpha: float) -> str:
        """Formata a resposta com apostas + rodapé informativo."""
        linhas = ["🎰 <b>SUAS APOSTAS INTELIGENTES</b> 🎰\n"]
        for i, aposta in enumerate(apostas, 1):
            pares = sum(1 for n in aposta if n % 2 == 0)
            linhas.append(
                f"<b>Aposta {i}:</b> {' '.join(f'{n:02d}' for n in aposta)}\n"
                f"🔢 Pares: {pares} | Ímpares: {15 - pares}\n"
            )
        if SHOW_TIMESTAMP:
            now_sp = datetime.now(ZoneInfo(TIMEZONE))
            carimbo = now_sp.strftime("%Y-%m-%d %H:%M:%S %Z")
            linhas.append(f"<i>janela={janela} | α={alpha:.2f} | {carimbo}</i>")
        return "\n".join(linhas)

    # ---------- Utilitários Mestre (baseado só no último resultado) ----------
    @staticmethod
    def _contar_pares(aposta):
        return sum(1 for n in aposta if n % 2 == 0)

    @staticmethod
    def _max_seq(aposta):
        """Maior sequência consecutiva (ex.: [7,8,9] = 3)."""
        s = sorted(aposta)
        best = cur = 1
        for i in range(1, len(s)):
            if s[i] == s[i-1] + 1:
                cur += 1
                best = max(best, cur)
            else:
                cur = 1
        return best

    @staticmethod
    def _complemento(last_set):
        return [n for n in range(1, 26) if n not in last_set]

    def _ajustar_paridade_e_seq(self, aposta, alvo_par=(7, 8), max_seq=3):
        """
        Ajusta determinísticamente a aposta para paridade 7–8 e máx. sequência 3,
        trocando com números do complemento (1..25 \ aposta).
        """
        aposta = sorted(set(aposta))
        comp = [n for n in range(1, 26) if n not in aposta]

        def tentar_quebrar_sequencias(a):
            changed = False
            while self._max_seq(a) > max_seq and comp:
                s = sorted(a)
                start = s[0]
                run_len = 1
                seqs = []
                for i in range(1, len(s)):
                    if s[i] == s[i-1] + 1:
                        run_len += 1
                    else:
                        if run_len > 1:
                            seqs.append((start, s[i-1], run_len))
                        start = s[i]
                        run_len = 1
                if run_len > 1:
                    seqs.append((start, s[-1], run_len))
                seqs.sort(key=lambda t: t[2], reverse=True)
                _, fim, _ = seqs[0]
                subs = None
                for c in comp:
                    if (c-1 not in a) and (c+1 not in a):  # tenta não criar nova sequência
                        subs = c
                        break
                if subs is None:
                    subs = comp[0]
                a.remove(fim)
                a.append(subs)
                comp.remove(subs)
                changed = True
                a.sort()
            return changed

        def tentar_ajustar_paridade(a):
            min_par, max_par = alvo_par
            pares = self._contar_pares(a)
            if pares > max_par:
                rem = next((x for x in a if x % 2 == 0), None)
                add = next((c for c in comp if c % 2 == 1), None)
            elif pares < min_par:
                rem = next((x for x in a if x % 2 == 1), None)
                add = next((c for c in comp if c % 2 == 0), None)
            else:
                return False
            if rem is not None and add is not None:
                a.remove(rem)
                a.append(add)
                comp.remove(add)
                a.sort()
                return True
            return False

        for _ in range(10):
            m1 = tentar_quebrar_sequencias(aposta)
            m2 = tentar_ajustar_paridade(aposta)
            if not m1 and not m2:
                break
        return sorted(aposta)

    def _construir_aposta_por_repeticao(self, last_sorted, comp_sorted, repeticoes, offset_last=0, offset_comp=0):
        """
        Monta uma aposta determinística com 'repeticoes' vindas do último resultado,
        completando com ausentes. Usa offsets para variar jogos de forma reprodutível.
        """
        L = last_sorted
        C = comp_sorted
        base = L[offset_last % 15:] + L[:offset_last % 15]
        manter = base[:repeticoes]
        faltam = 15 - len(manter)
        comp_rot = C[offset_comp % len(C):] + C[:offset_comp % len(C)]
        completar = comp_rot[:faltam]
        aposta = sorted(set(manter + completar))
        if len(aposta) < 15:
            for n in C:
                if n not in aposta:
                    aposta.append(n)
                if len(aposta) == 15:
                    break
        return sorted(aposta)

    # --------- Seed/Salt estável para personalizar o /mestre ---------
    @staticmethod
    def _stable_hash_int(texto: str) -> int:
        """Hash estável → inteiro, independente do processo."""
        return int(hashlib.blake2b(texto.encode("utf-8"), digest_size=8).hexdigest(), 16)

    def _calc_mestre_seed(self, user_id: int, chat_id: int, ultimo_sorted: list[int]) -> int:
        """
        Gera uma semente estável baseada no usuário, chat e último resultado.
        Assim, cada usuário/chat recebe um pacote diferente, mas reprodutível.
        """
        ultimo_str = "".join(f"{n:02d}" for n in ultimo_sorted)  # ex: "010203...25"
        key = f"{user_id}|{chat_id}|{ultimo_str}"
        return self._stable_hash_int(key)

    # --------- Diversificador do Mestre ---------
    def _diversificar_mestre(self, apostas, ultimo, comp, max_rep_ultimo=7, min_mid=3, min_fortes=2):
        """
        Aplica refinamentos determinísticos nas apostas geradas pelo Mestre:
        - Garante pelo menos 'min_fortes' dezenas de AUSENTES FORTES por aposta
        - Limita a repetição de cada dezena do último resultado a 'max_rep_ultimo'
        - Garante pelo menos 'min_mid' dezenas na faixa [12..18] em cada aposta
        Mantém paridade (7–8) e max_seq<=3 após cada ajuste.
        """
        from collections import Counter

        comp_set = set(comp)
        # 1) AUSENTES FORTES (ordem determinística)
        preferidos = [20, 22, 24, 10, 12, 14, 16, 18]
        ausentes_fortes = [n for n in preferidos if n in comp_set]

        for idx, a in enumerate(apostas):
            a = sorted(a)
            fortes_na_aposta = sum(1 for n in a if n in ausentes_fortes)
            if fortes_na_aposta < min_fortes and ausentes_fortes:
                faltam = min_fortes - fortes_na_aposta
                removiveis = sorted([x for x in a if x in ultimo], reverse=True)
                for _ in range(faltam):
                    add = next((c for c in ausentes_fortes if c not in a), None)
                    if add is None:
                        break
                    rem = None
                    for r in list(removiveis):
                        if r in a:
                            rem = r
                            removiveis.remove(r)
                            break
                    if rem is None:
                        rem = a[-1]
                    if rem == add:
                        continue
                    a.remove(rem)
                    a.append(add)
                    a.sort()
                a = self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3)
                apostas[idx] = a

        # 2) LIMITAR REPETIÇÃO de dezenas do último resultado
        cnt = Counter()
        for a in apostas:
            for n in a:
                if n in ultimo:
                    cnt[n] += 1

        excesso = [(n, cnt[n]) for n in sorted(ultimo, reverse=True) if cnt[n] > max_rep_ultimo]
        if excesso and comp_set:
            comp_ord = sorted(comp_set)
            for dezena, _ in excesso:
                for i in range(len(apostas) - 1, -1, -1):
                    if cnt[dezena] <= max_rep_ultimo:
                        break
                    a = apostas[i][:]
                    if dezena not in a:
                        continue
                    add = next((c for c in comp_ord if c not in a), None)
                    if add is None:
                        break
                    a.remove(dezena)
                    a.append(add)
                    a.sort()
                    a = self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3)
                    apostas[i] = a
                    cnt[dezena] -= 1

        # 3) GARANTIR FAIXA MÉDIA (12..18)
        mid_lo, mid_hi = 12, 18
        for i, a in enumerate(apostas):
            a = sorted(a)
            mid = [n for n in a if mid_lo <= n <= mid_hi]
            if len(mid) < min_mid:
                need = min_mid - len(mid)
                candidatos_add = [n for n in sorted(comp_set) if mid_lo <= n <= mid_hi and n not in a]
                if len(candidatos_add) < need:
                    extras = [n for n in range(mid_lo, mid_hi + 1) if n not in a]
                    for x in extras:
                        if x not in candidatos_add:
                            candidatos_add.append(x)
                candidatos_rem = [x for x in sorted(a, reverse=True) if not (mid_lo <= x <= mid_hi) and x in ultimo]
                if len(candidatos_rem) < need:
                    outros = [x for x in sorted(a, reverse=True) if not (mid_lo <= x <= mid_hi)]
                    for x in outros:
                        if x not in candidatos_rem:
                            candidatos_rem.append(x)
                j = 0
                while need > 0 and j < len(candidatos_add) and j < len(candidatos_rem):
                    add = candidatos_add[j]
                    rem = candidatos_rem[j]
                    if add == rem:
                        j += 1
                        continue
                    if rem in a and add not in a:
                        a.remove(rem)
                        a.append(add)
                        a.sort()
                        a = self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3)
                        need -= 1
                    j += 1
                apostas[i] = a

        return [sorted(a) for a in apostas]

    # --------- Gerador mestre (com seed por usuário/chat) ---------
    def _gerar_mestre_por_ultimo_resultado(self, historico, seed: int | None = None):
        """
        Gera 10 apostas determinísticas a partir do último resultado:
        - 1x com 8R
        - 1x com 11R
        - demais com 9–10R
        Regras: paridade 7–8 e max_seq=3, cobrindo ausentes.
        Personalizado por usuário/chat via seed (reprodutível).
        """
        ultimo = sorted(historico[-1])
        comp = self._complemento(set(ultimo))

        # plano de repetição (10 jogos): 10,10,9,9,10,9,10,8,11,10
        planos = [10, 10, 9, 9, 10, 9, 10, 8, 11, 10]

        # semente para offsets
        seed = int(seed or 0)

        apostas = []
        for i, r in enumerate(planos):
            # offsets derivados da seed (mantendo faixas válidas)
            off_last = (i + seed) % 15
            if len(comp) > 0:
                off_comp = (i * 2 + seed // 15) % len(comp)
            else:
                off_comp = 0

            aposta = self._construir_aposta_por_repeticao(
                last_sorted=ultimo,
                comp_sorted=comp,
                repeticoes=r,
                offset_last=off_last,
                offset_comp=off_comp,
            )
            aposta = self._ajustar_paridade_e_seq(aposta, alvo_par=(7, 8), max_seq=3)
            apostas.append(aposta)

        # cobertura de ausentes: se algum ausente não entrou em nenhuma aposta, force inclusão trocando da última aposta
        ausentes = set(comp)
        presentes_em_alguma = set(n for a in apostas for n in a)
        faltantes = [n for n in ausentes if n not in presentes_em_alguma]
        if faltantes:
            a = apostas[-1][:]
            for n in faltantes:
                subs_idx = next((idx for idx, x in enumerate(reversed(a)) if x in ultimo), None)
                if subs_idx is not None:
                    idx_real = len(a) - 1 - subs_idx
                    a[idx_real] = n
                    a.sort()
            a = self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3)
            apostas[-1] = a

        # Diversificação e reequilíbrio final
        apostas = self._diversificar_mestre(
            apostas, ultimo=ultimo, comp=set(comp),
            max_rep_ultimo=7, min_mid=3, min_fortes=2
        )

        return apostas

    # --- Novo comando: /mestre ---
    async def mestre(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Preset 'Mestre' baseado APENAS no último resultado do histórico.
        - Gera 10 apostas determinísticas (9R–10R + 1x 8R + 1x 11R)
        - Paridade 7–8 e máx. sequência = 3
        - Cobre ausentes ao longo do pacote
        - Personaliza por usuário/chat (seed reprodutível)
        """
        user_id = update.effective_user.id
        if not self._usuario_autorizado(user_id):
            await update.message.reply_text("⛔ Você não está autorizado a gerar apostas.")
            return

        # cooldown por chat
        chat_id = update.effective_chat.id
        if self._hit_cooldown(chat_id, "mestre"):
            await update.message.reply_text(f"⏳ Aguarde {COOLDOWN_SECONDS}s para usar /mestre novamente.")
            return

        # carrega histórico e pega somente o último resultado
        try:
            historico = carregar_historico(HISTORY_PATH)
            if not historico:
                await update.message.reply_text("Erro: histórico vazio.")
                return
        except Exception as e:
            await update.message.reply_text(f"Erro ao carregar histórico: {e}")
            return

        # seed personalizada por usuário/chat/último resultado
        try:
            ultimo_sorted = sorted(historico[-1])
            seed = self._calc_mestre_seed(
                user_id=update.effective_user.id,
                chat_id=update.effective_chat.id,
                ultimo_sorted=ultimo_sorted,
            )
        except Exception:
            seed = 0

        try:
            apostas = self._gerar_mestre_por_ultimo_resultado(historico, seed=seed)
        except Exception as e:
            logger.error("Erro no preset Mestre (último resultado):\n" + traceback.format_exc())
            await update.message.reply_text(f"Erro no preset Mestre: {e}")
            return

        # formatação
        linhas = ["🎰 <b>SUAS APOSTAS INTELIGENTES — Preset Mestre</b> 🎰\n"]
        for i, aposta in enumerate(apostas, 1):
            pares = self._contar_pares(aposta)
            linhas.append(
                f"<b>Aposta {i}:</b> {' '.join(f'{n:02d}' for n in aposta)}\n"
                f"🔢 Pares: {pares} | Ímpares: {15 - pares}\n"
            )
        if SHOW_TIMESTAMP:
            now_sp = datetime.now(ZoneInfo(TIMEZONE))
            carimbo = now_sp.strftime("%Y-%m-%d %H:%M:%S %Z")
            linhas.append(f"<i>base=último resultado | paridade=7–8 | max_seq=3 | {carimbo}</i>")

        await update.message.reply_text("\n".join(linhas), parse_mode="HTML")

    # --- Diagnóstico ---
    async def ping(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("pong")

    async def versao(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        txt = (
            f"🤖 Versão do bot\n"
            f"- BUILD_TAG: <code>{BUILD_TAG}</code>\n"
            f"- Import layout: <code>{LAYOUT}</code>\n"
            f"- Comandos: /start /gerar /mestre /meuid /autorizar /remover /backtest /ping /versao"
        )
        await update.message.reply_text(txt, parse_mode="HTML")

    # --- Comandos auxiliares (meuid, autorizar, remover) ---
    async def meuid(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        await update.message.reply_text(
            f"🆔 Seu ID: <code>{user_id}</code>\nUse este código para liberação.",
            parse_mode="HTML",
        )

    async def autorizar(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_user.id != self.admin_id:
            return await update.message.reply_text("⛔ Você não tem permissão.")
        if len(context.args) != 1 or not context.args[0].isdigit():
            return await update.message.reply_text("Uso: /autorizar <ID>")
        user_id = int(context.args[0])
        self.whitelist.add(user_id)
        self._salvar_whitelist()
        await update.message.reply_text(f"✅ Usuário {user_id} autorizado.")

    async def remover(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_user.id != self.admin_id:
            return await update.message.reply_text("⛔ Você não tem permissão.")
        if len(context.args) != 1 or not context.args[0].isdigit():
            return await update.message.reply_text("Uso: /remover <ID>")
        user_id = int(context.args[0])
        if user_id in self.whitelist:
            self.whitelist.remove(user_id)
            self._salvar_whitelist()
            await update.message.reply_text(f"✅ Usuário {user_id} removido.")
        else:
            await update.message.reply_text("ℹ️ Usuário não está na whitelist.")

    # ------------- Handler do backtest -------------
    async def backtest(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Comando oculto /backtest – apenas admin.
        Padrão: janela=30 | bilhetes=3 | α=0,55
        """
        user_id = update.effective_user.id
        if not self._is_admin(user_id):
            return
        janela, bilhetes_por_concurso, alpha = self._parse_backtest_args(context.args)
        await update.message.reply_text(
            f"Executando backtest com janela={janela}, bilhetes={bilhetes_por_concurso}, α={alpha:.2f}..."
        )
        loop = asyncio.get_running_loop()
        try:
            historico = carregar_historico(HISTORY_PATH)
            func = partial(
                executar_backtest_resumido,
                historico=historico,
                janela=janela,
                bilhetes_por_concurso=bilhetes_por_concurso,
                alpha=alpha,
            )
            resumo: str = await loop.run_in_executor(None, func)
            if len(resumo) > 4000:
                resumo = resumo[:4000] + "\n\n[Saída truncada]"
            await update.message.reply_text("📊 BACKTEST\n" + resumo)
        except Exception as e:
            logger.error("Erro no backtest:\n" + traceback.format_exc())
            await update.message.reply_text(f"Erro no backtest: {e}")

    def run(self):
        logger.info("Bot iniciado e aguardando comandos.")
        self.app.run_polling()

# ========================
# Execução
# ========================
if __name__ == "__main__":
    bot = LotoFacilBot()
    bot.run()





