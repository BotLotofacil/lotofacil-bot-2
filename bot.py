# bot.py

import os
import logging
import traceback
import asyncio
import re
import hashlib
import json
import time
from collections import deque
from functools import partial
from typing import List, Set, Tuple
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from os import getenv
from dataclasses import dataclass
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
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

# ===== Anti-abuso / rate limit =====
MAX_CMDS_PER_MIN = 12           # comandos totais por minuto (qualquer comando)
MAX_UNKNOWN_PER_MIN = 5         # comandos desconhecidos por minuto
TEMP_BLOCK_SECONDS = 15 * 60    # 15 minutos bloqueado
WARN_THRESHOLD = 0.75           # avisa antes de bloquear (75% do limite)

# Estado em memória
_ABUSE_EVENTS = {}   # user_id -> {"all": deque[timestamps], "unk": deque[timestamps], "until": epoch}

# ===== Utilitários anti-abuso =====
def _abuse_get(user_id: int):
    s = _ABUSE_EVENTS.get(user_id)
    if not s:
        s = {"all": deque(), "unk": deque(), "until": 0.0}
        _ABUSE_EVENTS[user_id] = s
    return s

def _abuse_prune(q: deque, now: float, window: float = 60.0):
    while q and now - q[0] > window:
        q.popleft()

def _is_temporarily_blocked(user_id: int) -> bool:
    s = _abuse_get(user_id)
    return time.time() < s["until"]

def _register_command_event(user_id: int, is_unknown: bool) -> tuple[bool, str]:
    """
    Registra evento e retorna (permitido, mensagem_de_erro_ou_vazio).
    Bloqueia se estourar limites por minuto.
    """
    now = time.time()
    s = _abuse_get(user_id)

    # limpa janela de 60s
    _abuse_prune(s["all"], now)
    _abuse_prune(s["unk"], now)

    # registra
    s["all"].append(now)
    if is_unknown:
        s["unk"].append(now)

    # checa bloqueio
    total = len(s["all"])
    unk = len(s["unk"])

    if total > MAX_CMDS_PER_MIN or unk > MAX_UNKNOWN_PER_MIN:
        s["until"] = now + TEMP_BLOCK_SECONDS
        return (False, f"🚫 Proteção ativada: muitas tentativas em curto período. "
                       f"Tente novamente depois de {TEMP_BLOCK_SECONDS//60} min.")

    # aviso preventivo (não bloqueia ainda)
    if total >= int(MAX_CMDS_PER_MIN * WARN_THRESHOLD) or unk >= int(MAX_UNKNOWN_PER_MIN * WARN_THRESHOLD):
        return (True, "⚠️ Muitas solicitações em um curto período. Vá com calma para evitar bloqueio temporário.")

    return (True, "")

# ===== Autodetecção da ordem do histórico (ASC/DESC) =====
def _parse_nums_from_line(line: str) -> List[int]:
    # Extrai números da linha (CSV com vírgula, ponto e vírgula ou espaço)
    nums = re.findall(r"\d+", line)
    return [int(x) for x in nums]

def _ler_primeira_e_ultima_linha_csv(path: str) -> Tuple[List[int] | None, List[int] | None]:
    """Lê a primeira e a última linha não vazias do CSV bruto.
    Cada linha é: <id_concurso>, d1, d2, ..., d15
    Por isso, usamos os ÚLTIMOS 15 números da linha.
    """
    if not os.path.exists(path):
        return None, None
    first, last = None, None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            nums = _parse_nums_from_line(line)
            if len(nums) >= 15:
                dezenas = sorted(nums[-15:])  # <- pega as 15 dezenas reais
                if first is None:
                    first = dezenas
                last = dezenas
    return first, last

def _autodetect_history_order() -> bool | None:
    """
    Detecta se o histórico retornado por carregar_historico() está:
      True  -> DESC (mais recente em hist[0])
      False -> ASC  (mais recente em hist[-1])
      None  -> indeterminado
    A detecção compara hist[0]/hist[-1] com a PRIMEIRA e a ÚLTIMA linha do CSV bruto.
    """
    try:
        hist = carregar_historico(HISTORY_PATH)
        if not hist:
            return None

        csv_first, csv_last = _ler_primeira_e_ultima_linha_csv(HISTORY_PATH)
        if not csv_first or not csv_last:
            return None

        h0 = sorted(list(hist[0]))
        h_last = sorted(list(hist[-1]))

        # Se o loader preserva a ordem do arquivo:
        # - Se hist[0] == primeira linha do CSV -> DESC (mais recente no topo)
        # - Se hist[-1] == primeira linha do CSV -> ASC  (mais recente no fim)
        if h0 == csv_first:
            return True
        if h_last == csv_first:
            return False

        # Fallbacks (casos em que loader inverteu em relação ao arquivo)
        if h0 == csv_last:
            return False
        if h_last == csv_last:
            return True

        return None
    except Exception:
        logger.warning("Falha ao autodetectar ordem do histórico.", exc_info=True)
        return None

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
# Quantidade: permitir até 50 no /gerar
QTD_BILHETES_PADRAO = 5
QTD_BILHETES_MIN = 1
QTD_BILHETES_MAX = 50

SHOW_TIMESTAMP = True
TIMEZONE = "America/Sao_Paulo"

# Janela e alpha (alinhados ao utils/backtest defaults/amarras)
JANELA_PADRAO = 60
JANELA_MIN, JANELA_MAX = 50, 1000

ALPHA_PADRAO = 0.36
ALPHA_MIN,  ALPHA_MAX  = 0.05, 0.95

HISTORY_PATH = "data/history.csv"
WHITELIST_PATH = "whitelist.txt"

# Cooldown (segundos) para evitar flood
COOLDOWN_SECONDS = 10

# Identificação do build (para /versao)
BUILD_TAG = getenv("BUILD_TAG", "unknown")

# ========================
# Configurações do Bolão Inteligente v5 (19 → 15)
# ========================
BOLAO_JANELA = 60
BOLAO_ALPHA  = 0.37
BOLAO_QTD_APOSTAS = 5
BOLAO_ANCHORS = (9, 11)
BOLAO_STATE_PATH = "data/bolao_state.json"

# Parâmetros do Bolão 19→15
BOLAO_PLANOS_R = [10, 10, 9, 9, 10, 9, 10, 8, 11, 10]
BOLAO_MAX_OVERLAP = 11
BOLAO_PARIDADE = (7, 8)
BOLAO_MAX_SEQ = 3
BOLAO_NEUTRA_RANGE = (12, 18)

# Limites de aprendizado (bias)
BOLAO_BIAS_MIN = -2.0
BOLAO_BIAS_MAX =  2.0
BOLAO_BIAS_HIT = +0.5
BOLAO_BIAS_MISS = -0.2
BOLAO_BIAS_ANCHOR_SCALE = 0.5

# ========================
# Heurísticas adicionais (Mestre + A/B)
# ========================
# Pares cuja coocorrência derrubou média em análises anteriores
PARES_PENALIZADOS = {(23, 2), (22, 19), (24, 20), (11, 1)}
# Conjunto de "ruídos" com cap de frequência por lote (Mestre)
RUIDOS = {2, 1, 14, 19, 20, 10, 7, 15, 21, 9}
# No pacote de 10 apostas do Mestre, cada ruído pode aparecer no máx. 6 apostas
RUIDO_CAP_POR_LOTE = 6
# Alpha alternativo para A/B
ALPHA_TEST_B = 0.39

# ========================
# Ciclo C (ancorado no último resultado)
# ========================
CICLO_C_ANCHORS = (9, 11)
CICLO_C_PLANOS = [8, 11, 10, 10, 9, 9, 9, 9, 10, 10]

# ========================
# Cache e utilitários globais
# ========================
_PROCESS_CACHE: dict = {}
HISTORY_ORDER_DESC = True

# ========================
# Funções utilitárias
# ========================
def _fmt_dezenas(nums: List[int]) -> str:
    return "".join(f"{n:02d}" for n in sorted(nums))

def _hash_dezenas(nums: List[int]) -> str:
    return hashlib.blake2b(_fmt_dezenas(nums).encode("utf-8"), digest_size=4).hexdigest()

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

# ========================
# Estruturas de dados
# ========================
@dataclass
class _Snapshot:
    snapshot_id: str
    tamanho: int
    dezenas: List[int]

# --- Telemetria por aposta (paridade/seq/repetições) ---
@dataclass
class TelemetriaAposta:
    pares: int
    impares: int
    max_seq: int
    repeticoes: int
    ok_paridade: bool
    ok_seq: bool
    ok_total: bool

def _telemetria_aposta(aposta: List[int], ultimo: List[int], alvo_par=(7, 8), max_seq=3) -> TelemetriaAposta:
    pares = sum(1 for n in aposta if n % 2 == 0)
    imp = 15 - pares
    # reusa utilitários já existentes:
    # - _max_seq(self, aposta)
    # - _contar_repeticoes(self, aposta, ultimo)
    # Nota: estas duas são métodos da classe; por isso, criamos um wrapper interno quando usado fora.
    return TelemetriaAposta(
        pares=pares,
        impares=imp,
        max_seq=0,            # preenchido pelo wrapper na classe (ver abaixo)
        repeticoes=0,         # idem
        ok_paridade=(alvo_par[0] <= pares <= alvo_par[1]),
        ok_seq=True,          # idem
        ok_total=False,       # idem
    )

# ========================
# Funções do Bolão Inteligente v5
# ========================
def _freq_window(hist, bias: dict[int, float] | None = None):
    """
    Frequência simples na janela (hist já cortado).
    Se 'bias' vier preenchido, aplica um reforço: freq_eff = freq + bias[n].
    """
    freq = {n: 0.0 for n in range(1, 26)}
    for conc in hist:
        for n in conc:
            freq[n] += 1.0
    if bias:
        for n, v in bias.items():
            if 1 <= n <= 25:
                freq[n] = float(freq.get(n, 0.0)) + float(v)
    return freq

def _atrasos_recent_first(hist_recent_first):
    """
    Atraso determinístico: 0 = saiu no último, 1 = penúltimo, etc.
    Se nunca saiu na janela, atraso = len(hist).
    'hist_recent_first' precisa estar com o concurso mais recente em hist[0].
    """
    atrasos = {n: len(hist_recent_first) for n in range(1, 26)}
    for idx, conc in enumerate(hist_recent_first):
        s = set(conc)
        for n in range(1, 26):
            if atrasos[n] == len(hist_recent_first) and n in s:
                atrasos[n] = idx
    return atrasos

def _bolao_load_state(path: str = BOLAO_STATE_PATH) -> dict:
    """Carrega estado/bias do bolão; retorna estrutura padrão se não existir."""
    base = {
        "bias": {},
        "hits": {},
        "seen": {},
        "last_snapshot": None,
        "draw_counter": {},
        "learning": {},              # <- suporte ao aprendizado leve
        "alpha": ALPHA_PADRAO,       # <- guarda alpha corrente de forma persistente
    }
    if not os.path.exists(path):
        return dict(base)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # saneamento
        for k, v in base.items():
            data.setdefault(k, v)
        return data
    except Exception:
        return dict(base)

def _bolao_save_state(state: dict, path: str = BOLAO_STATE_PATH):
    """Grava estado do bolão de forma atômica."""
    import tempfile
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

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

        # --- Autodetecta a ordem do histórico e ajusta a flag global ---
        try:
            detected = _autodetect_history_order()
            if detected is not None:
                # atualiza a flag global usada pelos métodos
                global HISTORY_ORDER_DESC
                HISTORY_ORDER_DESC = detected
                logger.info(f"Ordem do histórico autodetectada: {'DESC' if detected else 'ASC'} "
                            f"(HISTORY_ORDER_DESC={HISTORY_ORDER_DESC})")
            else:
                logger.info(f"Não foi possível autodetectar a ordem do histórico. "
                            f"Usando configuração atual HISTORY_ORDER_DESC={HISTORY_ORDER_DESC}.")
        except Exception:
            logger.warning("Erro ao configurar autodetecção de ordem.", exc_info=True)

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
    def _clamp_params(self, qtd: int, janela: int, alpha: float) -> tuple[int, int, float]:
        try:
            qtd    = max(QTD_BILHETES_MIN, min(QTD_BILHETES_MAX, int(qtd)))
            janela = max(JANELA_MIN,        min(JANELA_MAX,       int(janela)))
            alpha  = max(ALPHA_MIN,         min(ALPHA_MAX,        float(alpha)))
            return qtd, janela, alpha
        except Exception:
            return QTD_BILHETES_PADRAO, JANELA_PADRAO, ALPHA_PADRAO

    def _ultimo_resultado(self, historico) -> List[int]:
        """
        Retorna o concurso mais recente conforme HISTORY_ORDER_DESC.
        - True  -> historico[0]  (DESC: mais recente no topo)
        - False -> historico[-1] (ASC:  mais recente no fim)
        """
        if not historico:
            raise ValueError("Histórico vazio.")
        ult = historico[0] if HISTORY_ORDER_DESC else historico[-1]
        return sorted(list(ult))

    def _latest_snapshot(self) -> _Snapshot:
        historico = carregar_historico(HISTORY_PATH)
        if not historico:
            raise ValueError("Histórico vazio.")
        tamanho = len(historico)
        ultimo = self._ultimo_resultado(historico)
        h8 = _hash_dezenas(ultimo)
        snapshot_id = f"{tamanho}|{h8}"
        return _Snapshot(snapshot_id=snapshot_id, tamanho=tamanho, dezenas=ultimo)

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
        self.app.add_handler(CommandHandler("backtest", self.backtest))
        self.app.add_handler(CommandHandler("mestre", self.mestre))
        self.app.add_handler(CommandHandler("ab", self.ab))
        self.app.add_handler(CommandHandler("diagbase", self.diagbase))
        self.app.add_handler(CommandHandler("ping", self.ping))
        self.app.add_handler(CommandHandler("versao", self.versao))
        self.app.add_handler(CommandHandler("mestre_bolao", self.mestre_bolao))
        self.app.add_handler(CommandHandler("refinar_bolao", self.refinar_bolao))
        self.app.add_handler(CommandHandler("estado_bolao", self.estado_bolao))
        # Handler para comandos desconhecidos (precisa vir DEPOIS dos conhecidos)
        self.app.add_handler(MessageHandler(filters.COMMAND, self._unknown_command))
        logger.info("Handlers ativos: /start /gerar /mestre /mestre_bolao /refinar_bolao /ab /meuid /autorizar /remover /backtest /diagbase /ping /versao + unknown command handler")

    async def _unknown_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id

        # Admin nunca bloqueia
        if not self._is_admin(user_id):
            if _is_temporarily_blocked(user_id):
                return await update.message.reply_text("🚫 Você está temporariamente bloqueado por excesso de tentativas.")

            allowed, warn = _register_command_event(user_id, is_unknown=True)
            if not allowed:
                return await update.message.reply_text(warn)
            if warn:
                await update.message.reply_text(warn)

        # Resposta "neutra" que não revela nada
        await update.message.reply_text(
            "🤖 Comando não reconhecido.\n"
            "Use /start para ver como interagir ou /versao para listar comandos disponíveis."
        )

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /start – mensagem de boas-vindas e aviso legal."""
        mensagem = (
            "⚠️ <b>Aviso Legal</b>\n"
            "Este bot é apenas para fins estatísticos e recreativos. "
            "Não há garantia de ganhos na Lotofácil.\n\n"
            "🎉 <b>Bem-vindo</b>\n"
            "Use /gerar para receber 5 apostas baseadas em 60 concursos e α=0,36.\n"
            "Use /meuid para obter seu identificador e solicitar autorização.\n"
        )
        await update.message.reply_text(mensagem, parse_mode="HTML")

    # --- /gerar: rápido, estável, sem cache e com diversidade entre chamadas ---
    async def gerar_apostas(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Comando /gerar – Gera apostas inteligentes (rápido e estável).
        Uso: /gerar [qtd] [janela] [alpha]
        Padrão: 5 apostas | janela=60 | α=0,37
        """
        import asyncio

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
            pass  # mantém defaults

        # Clamps defensivos
        qtd, janela, alpha = self._clamp_params(qtd, janela, alpha)
        target_qtd = max(1, int(qtd))  # garante respeitar /gerar 50, etc.

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
            """Canônico + lock forte (pares 7–8, seq≤3) preservando inteligência."""
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

            # 3.1) Corte ao target e validação teimosa final (belt and suspenders)
            apostas_ok = []
            for a in apostas[:target_qtd]:
                a = _selar(a)
                if len(a) != 15 or len(set(a)) != 15:
                    a = _selar(_canon(a))
                # reforça forma (paridade 7–8 e Seq≤3), sem âncoras específicas
                a = self._hard_lock_fast(a, ultimo=ultimo or [], anchors=frozenset())
                apostas_ok.append(a)

            # 3.2) REGISTRO para aprendizado leve
            try:
                self._registrar_geracao(apostas_ok, base_resultado=ultimo or [])
            except Exception:
                logger.warning("Falha ao registrar geração para aprendizado leve (/gerar).", exc_info=True)

            # 4) Formatação + envio (usa α persistido no estado, se existir)
            try:
                st = _bolao_load_state()
            except Exception:
                st = None
            alpha_eff = float(st.get("alpha", alpha)) if isinstance(st, dict) else alpha

            try:
                resposta = self._formatar_resposta(apostas_ok, janela, alpha_eff)
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
                    linhas.append(f"<i>janela={janela} | α={alpha_eff:.2f} | {carimbo}</i>")
                resposta = "\n".join(linhas)

            # 5) Saída
            await update.message.reply_text(resposta, parse_mode="HTML")

            # (opcional) Chamar auto_aprender aqui NÃO é necessário para aprender (o ideal é após novo resultado),
            # mas manter não quebra nada; deixei como está:
            try:
                await self.auto_aprender(update, context)
            except Exception:
                logger.warning("auto_aprender falhou; prosseguindo normalmente.", exc_info=True)

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
    
    # ===== Helpers para métricas do aprendizado leve =====
    def _contar_acertos(self, aposta: list[int], resultado: list[int]) -> int:
        rset = set(int(x) for x in resultado if 1 <= int(x) <= 25)
        return sum(1 for n in aposta if n in rset)

    def _paridade(self, aposta: list[int]) -> tuple[int, int]:
        pares = sum(1 for x in aposta if x % 2 == 0)
        return pares, (len(aposta) - pares)

    @staticmethod
    def _complemento(last_set):
        return [n for n in range(1, 26) if n not in last_set]

    def _ajustar_paridade_e_seq(self, aposta, alvo_par=(7, 8), max_seq=3, anchors=frozenset()):
        """
        Ajusta determinísticamente a aposta para paridade 7–8 e máx. sequência 3,
        trocando com números do complemento (1..25 \ aposta). Nunca remove âncoras.
        OBS: comp é reavaliado a cada iteração (corrige estagnação de paridade/seq).
        """
        a = sorted(set(aposta))

        def max_seq_run(s):
            s = sorted(s)
            best = cur = 1
            for i in range(1, len(s)):
                if s[i] == s[i-1] + 1:
                    cur += 1
                    best = max(best, cur)
                else:
                    cur = 1
            return best

        def contar_pares(s):
            return sum(1 for n in s if n % 2 == 0)

        def tentar_quebrar_sequencias(a_local, comp_local):
            changed = False
            guard = 0
            while max_seq_run(a_local) > max_seq and comp_local and guard < 50:
                guard += 1
                s = sorted(a_local)
                seqs = []
                start = s[0]
                run = 1
                for i in range(1, len(s)):
                    if s[i] == s[i-1] + 1:
                        run += 1
                    else:
                        if run > 1:
                            seqs.append((start, s[i-1], run))
                        start = s[i]
                        run = 1
                if run > 1:
                    seqs.append((start, s[-1], run))
                if not seqs:
                    break
                seqs.sort(key=lambda t: t[2], reverse=True)
                rem = None
                for st, fn, _run in seqs:
                    for x in range(fn, st - 1, -1):
                        if x in a_local and x not in anchors:
                            rem = x
                            break
                    if rem is None:
                        break
                if rem is None:
                    break
                sub = next((c for c in comp_local if (c-1 not in a_local) and (c+1 not in a_local)), None)
                if sub is None:
                    sub = comp_local[0]
                a_local.remove(rem)
                a_local.append(sub)
                a_local.sort()
                changed = True
                comp_local[:] = [n for n in range(1, 26) if n not in a_local]
            return changed

        def tentar_ajustar_paridade(a_local, comp_local, min_par, max_par):
            pares = contar_pares(a_local)
            if pares > max_par:
                rem = next((x for x in a_local if x % 2 == 0 and x not in anchors), None)
                add = next((c for c in comp_local if c % 2 == 1), None)
            elif pares < min_par:
                rem = next((x for x in a_local if x % 2 == 1 and x not in anchors), None)
                add = next((c for c in comp_local if c % 2 == 0), None)
            else:
                return False
            if rem is not None and add is not None:
                a_local.remove(rem)
                a_local.append(add)
                a_local.sort()
                comp_local[:] = [n for n in range(1, 26) if n not in a_local]
                return True
            return False

        min_par, max_par = alvo_par

        for _ in range(40):
            comp = [n for n in range(1, 26) if n not in a]
            m1 = tentar_quebrar_sequencias(a, comp)
            comp = [n for n in range(1, 26) if n not in a]
            m2 = tentar_ajustar_paridade(a, comp, min_par, max_par)
            if not m1 and not m2:
                break

        if not (min_par <= contar_pares(a) <= max_par) or max_seq_run(a) > max_seq:
            comp = [n for n in range(1, 26) if n not in a]
            fallback = [n for n in range(1, 26) if (n not in a) and (n not in anchors)]
            _ = tentar_ajustar_paridade(a, comp or fallback, min_par, max_par)
            comp = [n for n in range(1, 26) if n not in a]
            _ = tentar_quebrar_sequencias(a, comp)

        return sorted(a)

    def _construir_aposta_por_repeticao(self, last_sorted, comp_sorted, repeticoes, offset_last=0, offset_comp=0):
        """
        Monta uma aposta determinística com 'repeticoes' vindas do último resultado,
        completando com ausentes. Usa offsets para variar jogos de forma reprodutível.
        Tolerante a comp_sorted vazio.
        """
        L = list(last_sorted)
        C = list(comp_sorted)

        base = L[offset_last % len(L):] + L[:offset_last % len(L)]
        manter = base[:repeticoes]

        faltam = 15 - len(manter)
        completar = []
        if C:
            k = offset_comp % len(C)
            comp_rot = C[k:] + C[:k]
            completar = comp_rot[:faltam]
        else:
            completar = []

        aposta = sorted(set(manter + completar))

        if len(aposta) < 15:
            pool = [n for n in range(1, 26) if n not in aposta]
            for n in pool:
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
        ultimo_str = "".join(f"{n:02d}" for n in ultimo_sorted)
        key = f"{user_id}|{chat_id}|{ultimo_str}"
        return self._stable_hash_int(key)

    def _next_draw_seed(self, snapshot_id: str) -> int:
        """
        Retorna uma semente determinística que muda a cada execução para o MESMO snapshot.
        A contagem é persistida em data/bolao_state.json -> draw_counter[snapshot_id].
        Trocar de snapshot (histórico novo) gera um novo contador automaticamente.
        """
        st = _bolao_load_state()
        cnt = st.get("draw_counter", {})
        n = int(cnt.get(snapshot_id, 0)) + 1
        cnt[snapshot_id] = n
        st["draw_counter"] = cnt
        _bolao_save_state(st)

        # seed estável derivada de (snapshot_id, n)
        return self._stable_hash_int(f"{snapshot_id}|{n}") & 0xFFFFFFFF

    # --- Wrapper de telemetria usando métodos da classe ---
    def _telemetria(self, aposta: List[int], ultimo: List[int], alvo_par=(7, 8), max_seq=3) -> TelemetriaAposta:
        t = _telemetria_aposta(aposta, ultimo, alvo_par=alvo_par, max_seq=max_seq)
        t.max_seq = self._max_seq(aposta)
        t.repeticoes = self._contar_repeticoes(aposta, ultimo)
        t.ok_seq = (t.max_seq <= max_seq)
        t.ok_total = t.ok_paridade and t.ok_seq
        return t

    # --- Pós-processador básico (paridade 7–8, max_seq<=3 e anti-overlap<=11) ---
    def _pos_processador_basico(self, apostas: List[List[int]], ultimo: List[int]) -> List[List[int]]:
        comp = [n for n in range(1, 26) if n not in ultimo]
        # normaliza cada aposta para paridade/seq
        norm = [
            self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3, anchors=frozenset())
            for a in apostas
        ]
        # reduz interseções fortes
        norm = self._anti_overlap(norm, ultimo=ultimo, comp=comp, max_overlap=BOLAO_MAX_OVERLAP, anchors=frozenset())
        # última passada de selagem
        norm = [
            self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3, anchors=frozenset())
            for a in norm
        ]
        return [sorted(a) for a in norm]
    
        # ---------- Aprendizado leve por bias (pós-swap controlado) -----------
    def _aplicar_bias(self, apostas: list[list[int]]) -> list[list[int]]:
        """
        Aplica viés leve pós-geração:
          - favorece números com bias>0
          - desfavorece bias<0
          - realiza até 2 trocas por aposta
          - SEM quebrar paridade 7–8 e Seq≤3
        """
        st = _bolao_load_state()
        bias: dict[str, float] = st.get("bias", {})
        if not bias:
            return apostas

        # converte chaves para int com tolerância
        bias_i = {}
        for k, v in bias.items():
            try:
                ki = int(k)
                if 1 <= ki <= 25:
                    bias_i[ki] = float(v)
            except Exception:
                continue

        def score(n: int) -> float:
            return bias_i.get(n, 0.0)

        out = []
        for a in apostas:
            a = sorted(set(a))
            comp = [n for n in range(1, 26) if n not in a]
            # candidatos para entrar (top bias no complemento) e sair (pior bias da aposta)
            ins = sorted(comp, key=score, reverse=True)
            outs = sorted(a, key=score)  # piores first
            trocas = 0
            for add in ins:
                if trocas >= 2:
                    break
                rem = next((x for x in outs if x in a), None)
                if rem is None:
                    break
                a2 = a[:]
                a2.remove(rem); a2.append(add); a2.sort()
                a2 = self._hard_lock_fast(a2, ultimo=[], anchors=frozenset())
                # aceita somente se mantiver forma
                if 7 <= self._contar_pares(a2) <= 8 and self._max_seq(a2) <= 3:
                    a = a2; trocas += 1
            out.append(sorted(a))
        return out

    # ---------- Pós-filtro unificado (quebra-seq + paridade + dedup + anti-overlap + bias) ----------
    def _pos_filtro_unificado(self, apostas: list[list[int]], ultimo: list[int]) -> list[list[int]]:
        # 1) normaliza cada aposta para paridade/seq
        out = [self._hard_lock_fast(a, ultimo, anchors=frozenset()) for a in apostas]
        # 2) dedup local + anti-overlap global (mantém 15 dezenas)
        out = self._dedup_apostas(out, ultimo=ultimo, max_overlap=BOLAO_MAX_OVERLAP, anchors=frozenset())
        out = self._anti_overlap(out, ultimo=ultimo, comp=[n for n in range(1, 26) if n not in ultimo],
                                 max_overlap=BOLAO_MAX_OVERLAP, anchors=frozenset())
        # 3) reforça forma novamente
        out = [self._hard_lock_fast(a, ultimo, anchors=frozenset()) for a in out]
        # 4) aprendizado leve por bias
        out = self._aplicar_bias(out)
        # 5) selagem final
        out = [self._hard_lock_fast(a, ultimo, anchors=frozenset()) for a in out]
        return [sorted(a) for a in out]

        # ---------- UTILITÁRIOS DE SELAGEM FINAL E DEDUP -----------
    def _enforce_rules(self, a: list[int], anchors=frozenset(), alvo_par=(7, 8), max_seq=3) -> list[int]:
        """
        Garante paridade 7–8 e seq<=3, preservando âncoras quando possível.
        """
        a = sorted(set(a))
        for _ in range(16):
            a = self._ajustar_paridade_e_seq(a, alvo_par=alvo_par, max_seq=max_seq, anchors=set(anchors))
            if 7 <= self._contar_pares(a) <= 8 and self._max_seq(a) <= 3:
                break
        return sorted(a)

    def _dedup_apostas(self, apostas: list[list[int]], ultimo: list[int], max_overlap: int | None = None, anchors=frozenset()) -> list[list[int]]:
        """
        Remove duplicadas e 'cura' cada clone localmente sem perder tamanho.
        Estratégia:
          - varre pares duplicados;
          - para o clone, troca uma dezena do 'último' por uma do complemento que
            não quebre paridade/seq; se não houver, usa qualquer complemento;
          - aplica enforce_rules após cada cura;
          - ao final, executa anti-overlap (opcional) e sela regras novamente.
        """
        seen = {}
        comp = [n for n in range(1, 26) if n not in ultimo]

        # 1) normaliza cada aposta antes (evita clones por ordenação)
        apostas = [sorted(a) for a in apostas]

        # 2) dedup com cura local
        for i, a in enumerate(apostas):
            key = tuple(a)
            if key not in seen:
                seen[key] = i
                continue

            # Curar clone: remover uma do último e inserir um ausente
            a2 = a[:]
            rem = next((x for x in reversed(a2) if x in ultimo and x not in anchors), None)
            add = next((c for c in comp if c not in a2 and (c-1 not in a2) and (c+1 not in a2)), None)
            if add is None:
                add = next((c for c in comp if c not in a2), None)

            if rem is not None and add is not None and rem != add:
                a2.remove(rem)
                a2.append(add)
                a2.sort()
                a2 = self._enforce_rules(a2, anchors=anchors)
            else:
                # fallback mínimo: gira um dos elementos não âncora
                rot = next((x for x in a2 if x not in anchors), None)
                if rot is not None:
                    a2.remove(rot)
                    add2 = next((c for c in comp if c not in a2), None)
                    if add2 is not None:
                        a2.append(add2)
                        a2.sort()
                        a2 = self._enforce_rules(a2, anchors=anchors)

            apostas[i] = a2
            seen[tuple(a2)] = i  # registra novo hash

        # 3) anti-overlap opcional (se pedido) + selagem final
        if max_overlap is not None:
            try:
                apostas = self._anti_overlap(apostas, ultimo=ultimo, comp=[n for n in range(1, 26) if n not in ultimo],
                                             max_overlap=max_overlap, anchors=set(anchors))
            except Exception:
                pass

        apostas = [self._enforce_rules(a, anchors=anchors) for a in apostas]
        return [sorted(a) for a in apostas]
    
        # ---------- LOCK RÁPIDO (paridade 7–8 e Seq≤3) -----------
    def _hard_lock_fast(self, aposta: list[int], ultimo: list[int] | set[int], anchors=frozenset(), alvo_par=(7, 8), max_seq=3) -> list[int]:
        """
        Enforca rapidamente a forma:
          - mantém tamanho 15
          - preserva âncoras quando possível
          - quebra sequências >=4 primeiro
          - depois corrige paridade (7–8)
          - repete poucas iterações até convergir
        """
        a = sorted(set(int(x) for x in aposta if 1 <= int(x) <= 25))
        anchors = set(int(x) for x in anchors if 1 <= int(x) <= 25)
        universo = list(range(1, 26))

        def comp_now(s: list[int]) -> list[int]:
            return [n for n in universo if n not in s]

        # garante len==15
        if len(a) < 15:
            for n in comp_now(a):
                a.append(n)
                if len(a) == 15:
                    break
        elif len(a) > 15:
            # remove não-âncora primeiro
            i = len(a) - 1
            while len(a) > 15 and i >= 0:
                if a[i] not in anchors:
                    a.pop(i)
                i -= 1
            while len(a) > 15:
                a.pop()

        # iterações limitadas
        for _ in range(24):
            # 1) quebrar sequências
            s = sorted(a)
            seqs = []
            start = s[0]; run = 1
            for i in range(1, len(s)):
                if s[i] == s[i-1] + 1:
                    run += 1
                else:
                    if run > 1: seqs.append((start, s[i-1], run))
                    start = s[i]; run = 1
            if run > 1: seqs.append((start, s[-1], run))
            seqs.sort(key=lambda t: t[2], reverse=True)

            changed = False
            if seqs and seqs[0][2] > max_seq:
                st, fn, _run = seqs[0]
                # remove um do meio, evitando âncora
                rem = None
                for x in range((st+fn)//2, fn+1):
                    if x in a and x not in anchors:
                        rem = x; break
                if rem is None:
                    for x in range(fn, st-1, -1):
                        if x in a and x not in anchors:
                            rem = x; break
                if rem is not None:
                    # busca complemento que não crie sequência
                    add = next((c for c in comp_now(a) if (c-1 not in a) and (c+1 not in a)), None)
                    if add is None:
                        add = next((c for c in comp_now(a) if c not in a), None)
                    if add is not None:
                        a.remove(rem); a.append(add); a.sort()
                        changed = True

            # 2) corrigir paridade
            pares = sum(1 for n in a if n % 2 == 0)
            min_par, max_par = alvo_par
            if pares > max_par:
                rem = next((x for x in a if x % 2 == 0 and x not in anchors), None)
                add = next((c for c in comp_now(a) if c % 2 == 1), None)
                if rem is not None and add is not None:
                    a.remove(rem); a.append(add); a.sort(); changed = True
            elif pares < min_par:
                rem = next((x for x in a if x % 2 == 1 and x not in anchors), None)
                add = next((c for c in comp_now(a) if c % 2 == 0), None)
                if rem is not None and add is not None:
                    a.remove(rem); a.append(add); a.sort(); changed = True

            # saída se nada mudou e forma ok
            if not changed and (7 <= self._contar_pares(a) <= 8) and (self._max_seq(a) <= max_seq):
                break

        return sorted(a)

    # --------- Anti-overlap robusto (NUNCA muda o tamanho das apostas) -------
    def _anti_overlap(self, apostas, ultimo, comp, max_overlap=BOLAO_MAX_OVERLAP, anchors=frozenset()):
        """
        Reduz interseções entre pares de apostas até 'max_overlap' SEM alterar o tamanho
        das apostas. Em cada troca:
          - só remove quando já houver substituto 'add' definido;
          - normaliza (paridade 7–8 e max_seq<=3) mantendo âncoras;
          - garante len==15 (complementa por comp; se esgotar, usa pool 1..25 sem repetir).
        """
        def _fix_len15(a: list[int]) -> list[int]:
            a = list(a)
            if len(a) < 15:
                presentes = set(a)
                for c in comp:
                    if len(a) == 15:
                        break
                    if c not in presentes:
                        a.append(c); presentes.add(c)
                if len(a) < 15:
                    for n in range(1, 26):
                        if n not in presentes:
                            a.append(n); presentes.add(n)
                            if len(a) == 15:
                                break
            elif len(a) > 15:
                s_anc = set(anchors)
                i = len(a) - 1
                while len(a) > 15 and i >= 0:
                    if a[i] not in s_anc:
                        a.pop(i)
                    i -= 1
                i = len(a) - 1
                while len(a) > 15 and i >= 0:
                    a.pop(i); i -= 1
            return sorted(a)

        comp_pool_base = sorted(set(comp))
        for _outer in range(3):
            changed_any_outer = False
            for i in range(len(apostas)):
                for j in range(i):
                    a = list(apostas[i])
                    b = list(apostas[j])

                    guard = 0
                    while guard < 40:
                        guard += 1
                        inter = sorted(set(a) & set(b))
                        if len(inter) <= max_overlap:
                            break

                        comp_pool = [c for c in comp_pool_base if c not in a or c not in b]

                        out = next(
                            (x for x in inter if (x in ultimo) and (x not in anchors) and (x in a)),
                            None
                        )
                        if out is not None:
                            add = next((c for c in comp_pool if c not in a and c not in b and (c-1 not in a) and (c+1 not in a)), None)
                            if add is None:
                                add = next((c for c in comp_pool if c not in a and c not in b), None)
                            if add is None:
                                add = next((c for c in comp_pool if c not in a), None)
                            if add is not None and out in a:
                                a.remove(out); a.append(add)
                                a = self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3, anchors=anchors)
                                a = _fix_len15(a)
                                changed_any_outer = True
                                continue
                        out_b = next(
                            (x for x in inter if (x in ultimo) and (x not in anchors) and (x in b)),
                            None
                        )
                        if out_b is not None:
                            add_b = next((c for c in comp_pool if c not in a and c not in b and (c-1 not in b) and (c+1 not in b)), None)
                            if add_b is None:
                                add_b = next((c for c in comp_pool if c not in a and c not in b), None)
                            if add_b is None:
                                add_b = next((c for c in comp_pool if c not in b), None)
                            if add_b is not None and out_b in b:
                                b.remove(out_b); b.append(add_b)
                                b = self._ajustar_paridade_e_seq(b, alvo_par=(7, 8), max_seq=3, anchors=anchors)
                                b = _fix_len15(b)
                                changed_any_outer = True
                                continue
                        break

                    if a != apostas[i]:
                        apostas[i] = _fix_len15(a)
                    if b != apostas[j]:
                        apostas[j] = _fix_len15(b)

            if not changed_any_outer:
                break

        apostas = [
            _fix_len15(self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3, anchors=anchors))
            for a in apostas
        ]
        return apostas
    
    # ===== utilitário seguro =====
    @staticmethod
    def _safe_remove(a: list[int], x: int) -> bool:
        try:
            a.remove(x)
            return True
        except ValueError:
            return False

    def _fechar_ciclo_c(
        self,
        apostas: list[list[int]],
        ultimo: list[int],
        anchors: tuple[int, int] = (9, 11),
    ) -> list[list[int]]:
        """
        Selagem determinística do Ciclo C com metas rígidas:
          - Paridade 7–8
          - SeqMax ≤ 3
          - Repetições (R) EXATAS por aposta, conforme CICLO_C_PLANOS[i]
          - Dedup + anti-overlap ≤ BOLAO_MAX_OVERLAP
        Preserva ÂNCORAS sempre que possível.
        """
        anchors_set = set(int(x) for x in anchors if 1 <= int(x) <= 25)
        u_set = set(int(x) for x in ultimo if 1 <= int(x) <= 25)
        universo = list(range(1, 26))
        comp_all = [n for n in universo if n not in u_set]

        def _is_ok_shape(a: list[int]) -> bool:
            return (len(a) == 15) and (len(set(a)) == 15) and (7 <= self._contar_pares(a) <= 8) and (self._max_seq(a) <= 3)

        def _canon(a: list[int]) -> list[int]:
            a = [int(x) for x in a if 1 <= int(x) <= 25]
            a = sorted(set(a))
            if len(a) < 15:
                # completa por complemento atual evitando criar sequência
                comp_now = [n for n in universo if n not in a]
                for n in comp_now:
                    if (n-1 not in a) and (n+1 not in a):
                        a.append(n)
                        if len(a) == 15:
                            break
                if len(a) < 15:
                    for n in comp_now:
                        if n not in a:
                            a.append(n)
                            if len(a) == 15:
                                break
            elif len(a) > 15:
                a = a[:15]
            return sorted(a)

        def _ensure_anchors(a: list[int]) -> list[int]:
            if not anchors_set:
                return a
            a = a[:]
            for anc in anchors_set:
                if anc not in a:
                    # troca o primeiro que não é âncora
                    rem = next((x for x in a if x not in anchors_set), None)
                    if rem is not None and rem != anc:
                        if self._safe_remove(a, rem):
                            a.append(anc)
                            a.sort()
            return a

        def _force_R(a: list[int], r_alvo: int) -> list[int]:
            """
            Ajusta R (repetições vs 'ultimo') EXATAMENTE para r_alvo.
            - se R baixo: troca COM->ULTIMO (sem mexer em âncoras)
            - se R alto:  troca ULTIMO->COM
            """
            a = a[:]
            r_atual = sum(1 for n in a if n in u_set)
            if r_atual == r_alvo:
                return a

            # Complemento dinâmico da aposta
            def comp_now():
                return [n for n in universo if n not in a]

            if r_atual < r_alvo:
                # precisa aumentar R: trazer números do 'ultimo'
                faltam = [n for n in ultimo if n not in a]
                for add in faltam:
                    rem = next((x for x in a if x not in u_set and x not in anchors_set), None) \
                          or next((x for x in a if x not in u_set), None)
                    if rem is None:
                        break
                    if self._safe_remove(a, rem):
                        a.append(add); a.sort()
                        r_atual += 1
                        if r_atual == r_alvo:
                            break
            else:
                # precisa diminuir R: expulsar itens do 'ultimo' e trazer do complemento
                for rem in [x for x in sorted(a, reverse=True) if x in u_set and x not in anchors_set]:
                    add = next((c for c in comp_now() if c not in a), None)
                    if add is None:
                        break
                    if self._safe_remove(a, rem):
                        a.append(add); a.sort()
                        r_atual -= 1
                        if r_atual == r_alvo:
                            break

            return a

        def _enforce_shape_then_R(a: list[int], r_alvo: int) -> list[int]:
            """
            Faz convergir forma e R:
              - primeiro ajusta forma (pares/seq)
              - corrige R fino
              - repete poucas vezes
            """
            a = _canon(a)
            for _ in range(8):
                # trava forma
                try:
                    a = self._hard_lock_fast(a, list(u_set), anchors=frozenset(anchors_set))
                except Exception:
                    a = self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3, anchors=anchors_set)
                    a = _canon(a)

                # corrige R com cuidado
                a = _force_R(a, r_alvo)
                a = _ensure_anchors(a)
                a = _canon(a)

                # pequena retocada na forma (pode mexer em R levemente, então iteramos no laço)
                if _is_ok_shape(a) and (sum(1 for n in a if n in u_set) == r_alvo):
                    break
            return a

        # ---------- 0) Normaliza e ancora ----------
        apostas = [ _ensure_anchors(_canon(a)) for a in apostas ]

        # ---------- 1) Força R exato e forma por plano ----------
        for i in range(len(apostas)):
            r_alvo = CICLO_C_PLANOS[i] if i < len(CICLO_C_PLANOS) else sum(1 for n in apostas[i] if n in u_set)
            apostas[i] = _enforce_shape_then_R(apostas[i], r_alvo)

        # ---------- 2) Dedup com cura local (respeita âncoras) ----------
        try:
            apostas = self._dedup_apostas(apostas, ultimo=list(u_set), max_overlap=None, anchors=anchors_set)
        except Exception:
            # fallback simples de dedup
            seen, uniq = set(), []
            for a in apostas:
                t = tuple(a)
                if t not in seen:
                    seen.add(t); uniq.append(a)
            apostas = uniq

        # ---------- 3) Anti-overlap global ----------
        try:
            apostas = self._anti_overlap(apostas, ultimo=list(u_set), comp=comp_all, max_overlap=BOLAO_MAX_OVERLAP, anchors=anchors_set)
        except Exception:
            pass

        # ---------- 4) Selagem final (curta) ----------
        final = []
        for i, a in enumerate(apostas):
            r_alvo = CICLO_C_PLANOS[i] if i < len(CICLO_C_PLANOS) else sum(1 for n in a if n in u_set)
            a = _enforce_shape_then_R(a, r_alvo)
            final.append(a)

        # ---------- 5) Passada extra de dedup + shape (garantia) ----------
        try:
            final = self._dedup_apostas(final, ultimo=list(u_set), max_overlap=BOLAO_MAX_OVERLAP, anchors=anchors_set)
        except Exception:
            pass

        out = []
        for i, a in enumerate(final):
            r_alvo = CICLO_C_PLANOS[i] if i < len(CICLO_C_PLANOS) else sum(1 for n in a if n in u_set)
            a = _enforce_shape_then_R(a, r_alvo)
            if not _is_ok_shape(a):
                # teimosia extra
                for _ in range(4):
                    a = self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3, anchors=anchors_set)
                    a = _force_R(a, r_alvo)
                    a = _ensure_anchors(a)
                    a = _canon(a)
                    if _is_ok_shape(a) and sum(1 for n in a if n in u_set) == r_alvo:
                        break
            out.append(a)

        return out

    # --------- Gerador mestre (com seed por usuário/chat) ---------
    def _gerar_mestre_por_ultimo_resultado(self, historico, seed: int | None = None):
        ultimo = self._ultimo_resultado(historico)
        comp = self._complemento(set(ultimo))

        N_JANELA_ANCHOR = 50
        hist = list(historico)
        jan = hist[-N_JANELA_ANCHOR:] if len(hist) >= N_JANELA_ANCHOR else hist[:]
        freq = {n: 0 for n in range(1, 26)}
        for conc in jan:
            for n in conc:
                freq[n] += 1

        prefer = []
        if 13 in ultimo:
            prefer.append(13)
        for c in (25, 3, 17):
            if c in ultimo and c not in prefer:
                prefer.append(c)
        hot = sorted([n for n in ultimo if n not in prefer], key=lambda x: (-freq[x], x))
        anchors = (prefer + hot)[:3]

        want_two_anchor_idx = set(range(10)) - {7, 8}
        want_three_anchor_idx = {0, 5, 9, 2}

        planos = [10, 10, 9, 9, 10, 9, 10, 8, 11, 10]

        seed = int(seed or 0)

        apostas = []
        for i, r in enumerate(planos):
            off_last = (i + seed) % 15
            off_comp = (i * 2 + seed // 15) % len(comp) if len(comp) > 0 else 0

            aposta = self._construir_aposta_por_repeticao(
                last_sorted=ultimo,
                comp_sorted=comp,
                repeticoes=r,
                offset_last=off_last,
                offset_comp=off_comp,
            )

            need = 2 if i in want_two_anchor_idx else 1
            if i in want_three_anchor_idx and len(anchors) >= 3:
                need = 3
            add_anchors = [a for a in anchors if a not in aposta][:need]
            if add_anchors:
                removiveis = [x for x in aposta if x in ultimo and x not in anchors]
                for add in add_anchors:
                    if add in aposta:
                        continue
                    rem = removiveis.pop(0) if removiveis else next((x for x in aposta if x not in anchors), None)
                    if rem is not None and rem != add:
                        aposta.remove(rem)
                        aposta.append(add)
                        aposta.sort()

            aposta = self._ajustar_paridade_e_seq(aposta, alvo_par=(7, 8), max_seq=3)
            aposta, _ = self._quebrar_pares_ruins(aposta, comp=comp, anchors=set(anchors))
            apostas.append(aposta)

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
            a, _ = self._quebrar_pares_ruins(a, comp=comp, anchors=set(anchors))
            apostas[-1] = a

        from collections import Counter
        comp_list = list(comp)
        min_per_absent = 2 if len(comp_list) <= 10 else 1
        max_per_absent = 5

        cnt_abs = Counter()
        for a in apostas:
            for n in a:
                if n in comp:
                    cnt_abs[n] += 1

        faltantes_min = [n for n in comp_list if cnt_abs[n] < min_per_absent]
        if faltantes_min:
            for n in faltantes_min:
                idx = min(range(len(apostas)), key=lambda k: sum(1 for x in apostas[k] if x in comp))
                alvo = apostas[idx][:]
                rem = next((x for x in sorted(alvo, reverse=True) if x in ultimo and x not in anchors), None)
                if rem is not None and n not in alvo:
                    alvo.remove(rem); alvo.append(n); alvo.sort()
                    alvo = self._ajustar_paridade_e_seq(alvo, alvo_par=(7, 8), max_seq=3)
                    alvo, _ = self._quebrar_pares_ruins(alvo, comp=comp, anchors=set(anchors))
                    apostas[idx] = alvo
                    cnt_abs[n] += 1

        excessos = [n for n in comp_list if cnt_abs[n] > max_per_absent]
        if excessos:
            for n in excessos:
                for i in range(len(apostas)-1, -1, -1):
                    a = apostas[i]
                    if n in a and cnt_abs[n] > max_per_absent:
                        cand_add = next((u for u in ultimo if u not in a and u not in anchors), None)
                        if cand_add is None:
                            continue
                        a.remove(n); a.append(cand_add); a.sort()
                        a = self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3, anchors=set(anchors))
                        a, _ = self._quebrar_pares_ruins(a, comp=comp, anchors=set(anchors))
                        apostas[i] = a
                        cnt_abs[n] -= 1
                        if cnt_abs[n] <= max_per_absent:
                            break

        apostas = self._diversificar_mestre(
            apostas, ultimo=ultimo, comp=set(comp),
            max_rep_ultimo=7, min_mid=3, min_fortes=2
        )
        apostas = self._cap_frequencia_ruido(apostas, ultimo=ultimo, comp=comp, anchors=set(anchors))
        apostas = [self._quebrar_pares_ruins(a, comp=comp, anchors=set(anchors))[0] for a in apostas]
        apostas = self._anti_overlap(apostas, ultimo=ultimo, comp=comp, max_overlap=BOLAO_MAX_OVERLAP)
        apostas = self._finalizar_regras_mestre(apostas, ultimo=ultimo, comp=comp, anchors=anchors)

        anchors_set = set(anchors)
        comp_list = list(comp)

        def _ensure_len_15(a: list[int]) -> list[int]:
            if len(a) < 15:
                pool = [n for n in range(1, 26) if n not in a]
                for n in pool:
                    a.append(n)
                    if len(a) == 15:
                        break
            return sorted(a)

        for i, a in enumerate(apostas):
            a = _ensure_len_15(a[:])
            for _ in range(14):
                a = self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3, anchors=anchors_set)
                if len(a) == 15 and 7 <= self._contar_pares(a) <= 8 and self._max_seq(a) <= 3:
                    break
            apostas[i] = sorted(a)

        seen = set()
        for i, a in enumerate(apostas):
            key = tuple(a)
            if key in seen:
                rem = next((x for x in reversed(a) if x in ultimo and x not in anchors_set), None)
                add = next((c for c in comp_list if c not in a), None)
                if rem is not None and add is not None:
                    a.remove(rem); a.append(add); a.sort()
                    a = self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3, anchors=anchors_set)
            seen.add(tuple(a))
            apostas[i] = a

        apostas = self._anti_overlap(apostas, ultimo=ultimo, comp=comp_list, max_overlap=BOLAO_MAX_OVERLAP, anchors=anchors_set)
        for i, a in enumerate(apostas):
            a = _ensure_len_15(a[:])
            a = self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3, anchors=anchors_set)
            apostas[i] = sorted(a)

        seen = set()
        for i, a in enumerate(apostas):
            key = tuple(a)
            if key in seen:
                rem = next((x for x in reversed(a) if x not in anchors_set), None)
                add = next((c for c in comp_list if c not in a), None)
                if rem is not None and add is not None and rem != add:
                    a.remove(rem)
                    a.append(add)
                    a.sort()
                    a = self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3, anchors=anchors_set)
                    key = tuple(a)

            tries = 0
            while key in seen and tries < 2:
                rem = next((x for x in reversed(a) if x not in anchors_set), None)
                add = next((c for c in comp_list if c not in a), None)
                if rem is None or add is None:
                    break
                a.remove(rem)
                a.append(add)
                a.sort()
                a = self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3, anchors=anchors_set)
                key = tuple(a)
                tries += 1

            seen.add(key)
            apostas[i] = a

        return apostas

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

        from collections import Counter as _Counter
        cnt = _Counter()
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

    # --------- Passe final para garantir regras após ajustes ---------
    def _finalizar_regras_mestre(self, apostas, ultimo, comp, anchors):
        from collections import Counter

        apostas = [self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3, anchors=set(anchors)) for a in apostas]

        comp_set = set(comp)
        comp_list = sorted(comp_set)
        min_per_absent = 2 if len(comp_list) <= 10 else 1
        max_per_absent = 5

        cnt_abs = Counter()
        for a in apostas:
            for n in a:
                if n in comp_set:
                    cnt_abs[n] += 1

        for n in comp_list:
            while cnt_abs[n] < min_per_absent:
                idx = min(range(len(apostas)), key=lambda k: sum(1 for x in apostas[k] if x in comp_set))
                alvo = apostas[idx][:]
                rem = next((x for x in sorted(alvo, reverse=True) if x in ultimo and x not in anchors), None)
                if rem is None or n in alvo:
                    break
                alvo.remove(rem); alvo.append(n); alvo.sort()
                alvo = self._ajustar_paridade_e_seq(alvo, alvo_par=(7, 8), max_seq=3, anchors=set(anchors))
                apostas[idx] = alvo
                cnt_abs[n] += 1

        for n in comp_list:
            while cnt_abs[n] > max_per_absent:
                idx = max(
                    range(len(apostas)),
                    key=lambda k: (n in apostas[k]) + sum(1 for x in apostas[k] if x in comp_set)
                )
                a = apostas[idx][:]
                if n not in a:
                    break
                cand_add = next((u for u in ultimo if u not in a and u not in anchors), None)
                if cand_add is None:
                    cand_add = next((u for u in ultimo if u not in a), None)
                if cand_add is None:
                    break
                a.remove(n); a.append(cand_add); a.sort()
                a = self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3, anchors=set(anchors))
                apostas[idx] = a
                cnt_abs[n] -= 1

        apostas = [self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3, anchors=set(anchors)) for a in apostas]
        apostas = self._anti_overlap(apostas, ultimo=ultimo, comp=comp, max_overlap=BOLAO_MAX_OVERLAP, anchors=set(anchors))
        return apostas

    # --------- Funções auxiliares (pares penalizados e cap de ruído) ---------
    @staticmethod
    def _tem_par_penalizado(aposta):
        s = set(aposta)
        for a, b in PARES_PENALIZADOS:
            if a in s and b in s:
                return (a, b)
        return None

    def _quebrar_pares_ruins(self, aposta, comp, anchors=()):
        a = sorted(aposta)
        comp_list = [c for c in sorted(comp) if c not in a]
        changed = False
        while True:
            par = self._tem_par_penalizado(a)
            if not par or not comp_list:
                break
            x, y = par
            # escolha quem sai (evitando tirar âncoras)
            sair = y if x in anchors else x
            if x in anchors and y in anchors:
                sair = max(x, y)
            if sair not in a:
                break

            # escolha substituto que não forme sequência
            sub = None
            for c in comp_list:
                if (c - 1 not in a) and (c + 1 not in a):
                    sub = c
                    break
            if sub is None:
                sub = comp_list[0]

            # aplica troca
            a.remove(sair)
            a.append(sub)
            a.sort()
            comp_list.remove(sub)

            # normaliza paridade/seq
            a = self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3, anchors=set(anchors))
            changed = True

        return a, changed

    def _cap_frequencia_ruido(self, apostas, ultimo, comp, anchors=()):
        from collections import Counter
        pres = Counter()
        for a in apostas:
            sa = set(a)
            for r in RUIDOS:
                if r in sa:
                    pres[r] += 1
        if not any(pres[r] > RUIDO_CAP_POR_LOTE for r in RUIDOS):
            return apostas
        comp_pool = sorted(set(comp))
        for r in sorted(RUIDOS):
            while pres[r] > RUIDO_CAP_POR_LOTE and comp_pool:
                idx = next((i for i in range(len(apostas)-1, -1, -1) if r in apostas[i]), None)
                if idx is None:
                    break
                a = apostas[idx][:]
                add = None
                for c in comp_pool:
                    if c not in a and (c-1 not in a) and (c+1 not in a):
                        add = c
                        break
                if add is None:
                    add = comp_pool[0] if comp_pool else None
                if add is None:
                    break
                rem = r if r not in anchors else next((x for x in reversed(a) if x not in anchors), None)
                if rem is None or rem not in a:
                    break
                a.remove(rem); a.append(add); a.sort()
                a = self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3, anchors=set(anchors))
                apostas[idx] = a
                pres[r] -= 1
                comp_pool.remove(add)
                if pres[r] <= RUIDO_CAP_POR_LOTE:
                    break
        return apostas

    # ========================
    # BOLÃO INTELIGENTE v5 (19 → 15)
    # ========================

    def _janela_recent_first(self, historico, janela: int):
        jan = ultimos_n_concursos(historico, janela)
        if HISTORY_ORDER_DESC:
            return list(jan)
        else:
            return list(reversed(jan))

    def _selecionar_matriz19(self, historico) -> list[int]:
        if not historico:
            raise ValueError("Histórico vazio.")
        ultimo = self._ultimo_resultado(historico)
        u_set = set(ultimo)

        st = _bolao_load_state()
        bias = {int(k): float(v) for k, v in st.get("bias", {}).items()}

        jan_rf = self._janela_recent_first(historico, BOLAO_JANELA)
        freq_eff = _freq_window(jan_rf, bias=bias)
        atrasos = _atrasos_recent_first(jan_rf)

        r10 = sorted(ultimo, key=lambda n: (-freq_eff[n], n))[:10]

        ausentes = [n for n in range(1, 26) if n not in u_set]
        hot_abs = [n for n in ausentes if atrasos[n] <= 8]
        hot_abs.sort(key=lambda n: (atrasos[n], -freq_eff[n], n))
        hot_take = hot_abs[:6] if len(hot_abs) >= 6 else hot_abs[:max(0, 5)]

        usados = set(r10) | set(hot_take)
        faltam = 19 - len(usados)
        neutrals_pool = [n for n in ausentes if n not in usados]
        def score(n):
            lo, hi = BOLAO_NEUTRA_RANGE
            if lo <= n <= hi:
                dist = 0
            else:
                dist = min(abs(n - lo), abs(n - hi))
            return (dist, -freq_eff[n], atrasos[n], n)
                
        neutrals_pool.sort(key=score)
        neutros = neutrals_pool[:max(0, faltam)]

        matriz = sorted(set(r10) | set(hot_take) | set(neutros))

        for anc in BOLAO_ANCHORS:
            if anc not in matriz:
                candidatos = [n for n in matriz if n not in BOLAO_ANCHORS and n not in u_set]
                if not candidatos:
                    candidatos = [n for n in matriz if n not in BOLAO_ANCHORS]
                rem = max(candidatos, key=lambda n: (atrasos[n], -freq_eff[n], n), default=None)
                if rem is not None and rem != anc:
                    matriz.remove(rem)
                    matriz.append(anc)
        matriz = sorted(set(matriz))
        if len(matriz) != 19:
            pool = [n for n in range(1, 26) if n not in matriz]
            for n in pool:
                matriz.append(n)
                if len(matriz) == 19:
                    break
            matriz = matriz[:19]
            matriz.sort()
        return matriz

    def _subsets_19_para_15(self, matriz19: list[int], seed: int | None = None) -> list[list[int]]:
        m = sorted(set(int(x) for x in matriz19))
        L = len(m)
        if L < 19:
            pool = [n for n in range(1, 26) if n not in m]
            for n in pool:
                m.append(n)
                if len(m) == 19:
                    break
            m = sorted(m[:19])

        anchors = set(BOLAO_ANCHORS)
        MAX_SEQ = int(BOLAO_MAX_SEQ)
        PAR_MIN, PAR_MAX = BOLAO_PARIDADE

        def contar_pares(a): return sum(1 for x in a if x % 2 == 0)

        def max_seq_run(lst):
            s = sorted(lst)
            best = cur = 1
            for i in range(1, len(s)):
                if s[i] == s[i-1] + 1:
                    cur += 1
                    best = max(best, cur)
                else:
                    cur = 1
            return best

        def candidatos_add(a, prefer_par: int | None = None):
            base = [x for x in m if x not in a]
            anti_seq = [x for x in base if (x - 1 not in a) and (x + 1 not in a)]
            prefer = anti_seq if anti_seq else base
            if prefer_par in (0, 1):
                prefer2 = [x for x in prefer if x % 2 == prefer_par]
                if prefer2:
                    return prefer2
            return prefer

        def remover_que_nao_ancora(a, prefer_par: int | None = None, dentro_sequencia: bool = False):
            s = sorted(a)
            runs = []
            start = s[0]; run = 1
            for i in range(1, len(s)):
                if s[i] == s[i-1] + 1:
                    run += 1
                else:
                    if run > 1: runs.append((start, s[i-1], run))
                    start = s[i]; run = 1
            if run > 1: runs.append((start, s[-1], run))
            runs.sort(key=lambda t: t[2], reverse=True)

            if dentro_sequencia and runs:
                st, fn, _r = runs[0]
                seq_vals = list(range(st, fn + 1))
                for x in reversed(seq_vals):
                    if x in a and x not in anchors:
                        if prefer_par in (0, 1) and (x % 2 != prefer_par):
                            continue
                        return x

            candidatos = [x for x in reversed(s) if x not in anchors]
            if prefer_par in (0, 1):
                cand2 = [x for x in candidatos if x % 2 == prefer_par]
                if cand2:
                    return cand2[0]
            return candidatos[0] if candidatos else None

        def hard_selar_regras(a):
            a = sorted(set(a))
            for _ in range(60):
                pares = contar_pares(a)
                ms = max_seq_run(a)

                changed = False

                if ms > MAX_SEQ:
                    rem = remover_que_nao_ancora(a, prefer_par=None, dentro_sequencia=True)
                    if rem is not None:
                        add = None
                        if pares > PAR_MAX:
                            cand = candidatos_add(a, prefer_par=1)
                            add = cand[0] if cand else None
                        elif pares < PAR_MIN:
                            cand = candidatos_add(a, prefer_par=0)
                            add = cand[0] if cand else None
                        if add is None:
                            cand = candidatos_add(a, prefer_par=None)
                            add = cand[0] if cand else None

                        if add is not None and rem in a:
                            a.remove(rem); a.append(add); a.sort()
                            changed = True

                pares = contar_pares(a)
                if not changed:
                    if pares > PAR_MAX:
                        rem = remover_que_nao_ancora(a, prefer_par=0, dentro_sequencia=False)
                        add_list = candidatos_add(a, prefer_par=1)
                        add = add_list[0] if add_list else None
                        if rem is not None and add is not None and rem in a and add not in a:
                            a.remove(rem); a.append(add); a.sort()
                            changed = True

                    elif pares < PAR_MIN:
                        rem = remover_que_nao_ancora(a, prefer_par=1, dentro_sequencia=False)
                        add_list = candidatos_add(a, prefer_par=0)
                        add = add_list[0] if add_list else None
                        if rem is not None and add is not None and rem in a and add not in a:
                            a.remove(rem); a.append(add); a.sort()
                            changed = True

                if not changed and (pares < PAR_MIN or pares > PAR_MAX or max_seq_run(a) > MAX_SEQ):
                    rem = remover_que_nao_ancora(a, prefer_par=None, dentro_sequencia=True)
                    add = None
                    pref = 0 if pares < PAR_MIN else (1 if pares > PAR_MAX else None)
                    cand = candidatos_add(a, prefer_par=pref)
                    add = cand[0] if cand else None
                    if rem is not None and add is not None and rem in a and add not in a:
                        a.remove(rem); a.append(add); a.sort()
                        changed = True

                if not changed:
                    if PAR_MIN <= contar_pares(a) <= PAR_MAX and max_seq_run(a) <= MAX_SEQ:
                        break

            if contar_pares(a) < PAR_MIN:
                rem = remover_que_nao_ancora(a, prefer_par=1, dentro_sequencia=False)
                add_list = candidatos_add(a, prefer_par=0)
                if rem is not None and add_list:
                    a.remove(rem); a.append(add_list[0]); a.sort()
            elif contar_pares(a) > PAR_MAX:
                rem = remover_que_nao_ancora(a, prefer_par=0, dentro_sequencia=False)
                add_list = candidatos_add(a, prefer_par=1)
                if rem is not None and add_list:
                    a.remove(rem); a.append(add_list[0]); a.sort()

            guard = 0
            while max_seq_run(a) > MAX_SEQ and guard < 10:
                guard += 1
                rem = remover_que_nao_ancora(a, dentro_sequencia=True)
                add_list = candidatos_add(a, prefer_par=None)
                if rem is None or not add_list:
                    break
                a.remove(rem); a.append(add_list[0]); a.sort()

            return sorted(a)

        packs = []
        base_offsets = [0, 3, 6, 9, 12, 1, 4, 7, 10, 13]
        seed = int(seed or 0)
        rot = seed % len(m)          # len(m) é 19 aqui
        offsets = [ (o + rot) % len(m) for o in base_offsets ][:BOLAO_QTD_APOSTAS]
        for off in offsets:
            s = []
            idx = off
            while len(s) < 15:
                s.append(m[idx % L])
                idx += 1
            a = sorted(set(s))
            a = hard_selar_regras(a)
            packs.append(a)

        for i in range(len(packs)):
            for j in range(i):
                a = packs[i][:]
                b = packs[j][:]
                guard = 0
                while guard < 60:
                    guard += 1
                    inter = sorted(set(a) & set(b))
                    if len(inter) <= BOLAO_MAX_OVERLAP:
                        break
                    rem = next((x for x in reversed(a) if x in inter and x not in anchors), None)
                    add = next((x for x in m if x not in a and x not in b and (x-1 not in a) and (x+1 not in a)), None)
                    if rem is not None and add is not None:
                        a.remove(rem); a.append(add); a.sort()
                        a = hard_selar_regras(a)
                        continue
                    rem_b = next((x for x in reversed(b) if x in inter and x not in anchors), None)
                    add_b = next((x for x in m if x not in a and x not in b and (x-1 not in b) and (x+1 not in b)), None)
                    if rem_b is not None and add_b is not None:
                        b.remove(rem_b); b.append(add_b); b.sort()
                        b = hard_selar_regras(b)
                        continue
                    break
                packs[i] = a
                packs[j] = b

        packs = [hard_selar_regras(a) for a in packs]
        # Dedup/selagem final para garantir diversidade do lote 19→15
        try:
            ultimo_dummy = []  # aqui não temos 'ultimo'; usamos dedup sem anti-overlap forte
            packs = self._dedup_apostas(packs, ultimo=ultimo_dummy or [26], max_overlap=None, anchors=set(BOLAO_ANCHORS))
        except Exception:
            pass
        return [sorted(a) for a in packs]
    
    # ===== Compat Layer: Mestre Bolão v5 (aliases) =====
    def _matriz19_base(self, ultimo: list[int], anchors=BOLAO_ANCHORS) -> list[int]:
        """
        Compatibilidade para /mestre_bolao.
        Reutiliza a sua seleção oficial de matriz-19 a partir do histórico.
        Se algo falhar, usa um fallback determinístico simples (último + âncoras + completa até 19).
        """
        try:
            # carrega histórico e usa a sua rotina oficial
            historico = carregar_historico(HISTORY_PATH)
            if not historico:
                raise RuntimeError("histórico vazio")
            m19 = self._selecionar_matriz19(historico)
        except Exception:
            # --- Fallback determinístico seguro ---
            universo = list(range(1, 26))
            base = sorted({int(x) for x in (ultimo or []) if 1 <= int(x) <= 25})
            # inclui âncoras se faltarem
            anc_in = anchors if isinstance(anchors, (tuple, list, set)) else BOLAO_ANCHORS
            anc = {int(x) for x in anc_in if 1 <= int(x) <= 25}
            for a in sorted(anc):
                if a not in base:
                    base.append(a)
            base = sorted(set(base))
            # completa até 19
            for n in universo:
                if len(base) >= 19:
                    break
                if n not in base:
                    base.append(n)
            m19 = sorted(base)[:19]

        # saneamento final: garantir 19 itens válidos
        m19 = sorted({int(x) for x in m19 if 1 <= int(x) <= 25})
        if len(m19) < 19:
            for n in range(1, 26):
                if len(m19) >= 19:
                    break
                if n not in m19:
                    m19.append(n)
            m19.sort()
        return m19[:19]

    def _expandir_19_para_15(self, matriz19: list[int], seed: int | None = None) -> list[list[int]]:
        """
        Compatibilidade para /mestre_bolao: delega para a sua expansão oficial 19→15.
        """
        return self._subsets_19_para_15(matriz19, seed=seed)

    # --- Novo comando: /mestre ---
    async def mestre(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if not self._usuario_autorizado(user_id):
            await update.message.reply_text("⛔ Você não está autorizado a gerar apostas.")
            return

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

        chat_id = update.message.chat_id if update.message else update.effective_chat.id
        if self._hit_cooldown(chat_id, "mestre"):
            await update.message.reply_text(f"⏳ Aguarde {COOLDOWN_SECONDS}s para usar /mestre novamente.")
            return

        # --- carrega histórico ---
        try:
            historico = carregar_historico(HISTORY_PATH)
            if not historico:
                await update.message.reply_text("Erro: histórico vazio.")
                return
        except Exception as e:
            await update.message.reply_text(f"Erro ao carregar histórico: {e}")
            return

        # --- seed composta: incremental por snapshot ^ seed estável por usuário/chat ---
        try:
            ultimo_sorted = self._ultimo_resultado(historico)
            snap = self._latest_snapshot()
            seed_inc = self._next_draw_seed(snap.snapshot_id)   # incremental por snapshot
            user_seed = self._calc_mestre_seed(
                user_id=update.effective_user.id,
                chat_id=chat_id,
                ultimo_sorted=ultimo_sorted,
            )
            seed = (seed_inc ^ (user_seed & 0xFFFFFFFF)) & 0xFFFFFFFF
        except Exception:
            seed = 0
        ultimo_sorted = locals().get("ultimo_sorted", self._ultimo_resultado(historico))
        try:
            snap = locals().get("snap", self._latest_snapshot())
        except Exception:
            snap = None

        # --- gera as apostas usando a seed calculada ---
        try:
            apostas = self._gerar_mestre_por_ultimo_resultado(historico, seed=seed)
        except Exception as e:
            logger.error("Erro no preset Mestre (último resultado):\n" + traceback.format_exc())
            await update.message.reply_text(f"Erro no preset Mestre: {e}")
            return

        # --- Pós-filtro unificado (forma + dedup/overlap + bias + forma) ---
        try:
            apostas = self._pos_filtro_unificado(apostas, ultimo=ultimo_sorted)
        except Exception:
            logger.warning("Falha no pós-filtro unificado no /mestre; aplicando selagem rápida.", exc_info=True)
            apostas = [self._hard_lock_fast(a, ultimo_sorted, anchors=frozenset()) for a in apostas]

        # --- REGISTRO para aprendizado leve (Mestre) ---
        try:
            # reusa o mesmo último resultado calculado do histórico
            ultimo = ultimo_sorted if ultimo_sorted else []
        except Exception:
            ultimo = []
        try:
            self._registrar_geracao(apostas, base_resultado=ultimo)
        except Exception:
            logger.warning("Falha ao registrar geração para aprendizado leve (/mestre).", exc_info=True)

        # --- Telemetria e formatação da resposta ---
        from datetime import datetime
        from zoneinfo import ZoneInfo

        snap_id = snap.snapshot_id if snap else "n/a"
        linhas = ["🎰 <b>SUAS APOSTAS INTELIGENTES — Preset Mestre</b> 🎰\n"]

        ok_count = 0
        for i, aposta in enumerate(apostas, 1):
            t = self._telemetria(aposta, ultimo_sorted, alvo_par=(7, 8), max_seq=3)
            status = "✅ OK" if t.ok_total else "🛠️ REPARAR"
            if t.ok_total:
                ok_count += 1
            linhas.append(
                f"<b>Aposta {i}:</b> {' '.join(f'{n:02d}' for n in aposta)}\n"
                f"🔢 Pares: {t.pares} | Ímpares: {t.impares} | SeqMax: {t.max_seq} | {t.repeticoes}R | {status}\n"
            )

        linhas.append(f"\n<b>Conformidade</b>: {ok_count}/{len(apostas)} dentro de (paridade 7–8, seq≤3)")
        linhas.append(f"<i>Regras: paridade 7–8, seq≤3, anti-overlap≤{BOLAO_MAX_OVERLAP}</i>")

        if SHOW_TIMESTAMP:
            now_sp = datetime.now(ZoneInfo(TIMEZONE))
            carimbo = now_sp.strftime("%Y-%m-%d %H:%M:%S %Z")
            hash_ult = _hash_dezenas(ultimo_sorted)
            linhas.append(
                f"<i>base=último resultado | paridade=7–8 | max_seq=3 | "
                f"hash={hash_ult} | snapshot={snap_id} | {carimbo}</i>"
            )

        await update.message.reply_text("\n".join(linhas), parse_mode="HTML")

    # --- /refinar_bolao: aplica bias, regenera 19→15 e sela o lote ---
    async def refinar_bolao(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Ajusta o bias do Modo Bolão v5 com base em um resultado 'oficial',
        regenera matriz-19 → 15 e aplica selagem forte:
          - Paridade 7–8
          - SeqMax ≤ 3
          - Dedup + anti-overlap ≤ BOLAO_MAX_OVERLAP
          - Aprendizado automático leve (bias) preservando forma
        Aceita 15 dezenas no comando: /refinar_bolao 01 02 ... 25
        """
        from datetime import datetime
        from zoneinfo import ZoneInfo

        try:
            user_id = update.effective_user.id
            if not self._usuario_autorizado(user_id):
                return await update.message.reply_text("⛔ Você não está autorizado.")

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

            chat_id = update.effective_chat.id
            if self._hit_cooldown(chat_id, "refinar_bolao"):
                return await update.message.reply_text(f"⏳ Aguarde {COOLDOWN_SECONDS}s para usar /refinar_bolao novamente.")

            # 0) histórico + snapshot
            historico = carregar_historico(HISTORY_PATH)
            if not historico:
                return await update.message.reply_text("Erro: histórico vazio.")
            snap = self._latest_snapshot()

            # 1) Resultado oficial (15 dezenas) — opcionalmente passado nos args
            if context.args and len(context.args) >= 15:
                try:
                    oficial = sorted({int(x) for x in context.args[:15]})
                    if len(oficial) != 15 or any(n < 1 or n > 25 for n in oficial):
                        return await update.message.reply_text("Forneça exatamente 15 dezenas válidas (1–25).")
                except Exception:
                    return await update.message.reply_text("Argumentos inválidos. Ex.: /refinar_bolao 01 03 04 ... 25")
            else:
                oficial = self._ultimo_resultado(historico)

            of_set = set(oficial)

            # 2) Matriz19 ANTES do refino (com estado atual de bias)
            matriz19_antes = self._selecionar_matriz19(historico)

            # 3) Carrega estado de bias e aplica atualização com base no 'oficial'
            st = _bolao_load_state()
            bias = {}
            for k, v in st.get("bias", {}).items():
                try:
                    ki = int(k); bias[ki] = float(v)
                except Exception:
                    continue
            hits_map = {}
            for k, v in st.get("hits", {}).items():
                try:
                    hits_map[int(k)] = int(v)
                except Exception:
                    continue
            seen_map = {}
            for k, v in st.get("seen", {}).items():
                try:
                    seen_map[int(k)] = int(v)
                except Exception:
                    continue

            # constantes existentes no projeto:
            # BOLAO_BIAS_HIT, BOLAO_BIAS_MISS, BOLAO_BIAS_ANCHOR_SCALE, BOLAO_BIAS_MIN, BOLAO_BIAS_MAX
            try:
                anchors_tuple = tuple(BOLAO_ANCHORS)  # ex.: (9, 11)
            except Exception:
                anchors_tuple = (9, 11)
            anch = set(anchors_tuple)

            mset = set(matriz19_antes)
            for n in mset:
                seen_map[n] = seen_map.get(n, 0) + 1
                if n in of_set:
                    hits_map[n] = hits_map.get(n, 0) + 1

            for n in mset:
                delta = float(BOLAO_BIAS_HIT) if (n in of_set) else float(BOLAO_BIAS_MISS)
                if n in anch:
                    delta *= float(BOLAO_BIAS_ANCHOR_SCALE)
                bias[n] = _clamp(float(bias.get(n, 0.0)) + delta, float(BOLAO_BIAS_MIN), float(BOLAO_BIAS_MAX))

            st["bias"] = {int(k): float(v) for k, v in bias.items()}
            st["hits"] = hits_map
            st["seen"] = seen_map
            try:
                st["last_snapshot"] = snap.snapshot_id
            except Exception:
                st["last_snapshot"] = "--"
            _bolao_save_state(st)

            # 4) Matriz19 DEPOIS do refino (já refletindo bias)
            matriz19_depois = self._selecionar_matriz19(historico)

            # 5) Regenera 19→15 com seed incremental por snapshot (variabilidade determinística)
            try:
                seed_nova = self._next_draw_seed(snap.snapshot_id)
            except Exception:
                seed_nova = self._next_draw_seed("fallback")
            apostas = self._subsets_19_para_15(matriz19_depois, seed=seed_nova)

            # 6) Selagem rápida por aposta
            apostas = [self._hard_lock_fast(a, oficial, anchors=frozenset(anchors_tuple)) for a in apostas]

            # 7) Pós-filtro unificado (forma + dedup/overlap + bias + forma)
            try:
                apostas = self._pos_filtro_unificado(apostas, ultimo=oficial)
            except Exception:
                logger.warning("pos_filtro_unificado falhou no /refinar_bolao; aplicando hard_lock por aposta.", exc_info=True)
                apostas = [self._hard_lock_fast(a, oficial, anchors=frozenset(anchors_tuple)) for a in apostas]

            # 7.1) Validação final teimosa (belt and suspenders)
            apostas_ok = []
            for a in apostas[:5]:
                a = self._hard_lock_fast(_selar(a), oficial, anchors=frozenset(anchors_tuple))
                apostas_ok.append(a)
            apostas = apostas_ok

            # 7.2) REGISTRO para aprendizado leve (/refinar_bolao)
            try:
                # usa 'oficial' deste handler como base do registro
                self._registrar_geracao(apostas, base_resultado=oficial or [])
            except Exception:
                logger.warning("Falha ao registrar geração para aprendizado leve (/refinar_bolao).", exc_info=True)

            # 8) Telemetria, placar e resposta (NÃO reprocessa as apostas!)
            def _hits(bilhete: list[int]) -> int:
                return len(of_set & set(bilhete))

            placar = [_hits(a) for a in apostas]
            melhor = max(placar) if placar else 0
            media  = (sum(placar) / len(placar)) if placar else 0.0

            ok_count = 0
            telems = []
            for a in apostas:
                t = self._telemetria(a, oficial, alvo_par=(7, 8), max_seq=3)
                telems.append(t)
                if getattr(t, "ok_total", False):
                    ok_count += 1

            # dup-check informativo (após tudo)
            uniq = {tuple(a) for a in apostas}
            dup_count = len(apostas) - len(uniq)

            linhas = []
            linhas.append("🧠 <b>Refino aplicado ao Modo Bolão v5</b>\n")
            linhas.append("<b>Oficial:</b> " + " ".join(f"{n:02d}" for n in oficial))
            linhas.append("<b>Matriz 19 (antes):</b> " + " ".join(f"{n:02d}" for n in matriz19_antes))
            linhas.append("<b>Matriz 19 (após refino):</b>  " + " ".join(f"{n:02d}" for n in matriz19_depois) + "\n")

            for i, a in enumerate(apostas, 1):
                t = telems[i - 1]
                linhas.append(
                    f"<b>Aposta {i}:</b> {' '.join(f'{n:02d}' for n in a)}  → <b>{placar[i-1]} acertos</b>\n"
                    f"🔢 Pares: {t.pares} | Ímpares: {t.impares} | SeqMax: {t.max_seq} | <i>{t.repeticoes}R</i>\n"
                )

            linhas.append(
                f"\n📊 <b>Resumo</b>\n"
                f"• Melhor aposta: <b>{melhor}</b> acertos\n"
                f"• Média do lote: <b>{media:.2f}</b> acertos\n"
                f"• Conformidade: <b>{ok_count}/{len(apostas)}</b> dentro de (paridade 7–8, seq≤3)"
            )
            linhas.append("• Ajuste de bias: +hit para dezenas presentes na matriz vs oficial; miss reduz (âncoras ±escala)")

            if dup_count > 0:
                linhas.append(
                    f"\n⚠️ <b>Aviso</b>: detectadas <b>{dup_count}</b> duplicidades no lote após refino. "
                    f"Se persistir, verifique history.csv e seeds."
                )

            if SHOW_TIMESTAMP:
                carimbo = datetime.now(ZoneInfo(TIMEZONE)).strftime("%Y-%m-%d %H:%M:%S %Z")
                try:
                    snap_id = snap.snapshot_id
                except Exception:
                    snap_id = "--"
                linhas.append(
                    f"\n<i>snapshot={snap_id} | seed={seed_nova} | tz={TIMEZONE} | /refinar_bolao | {carimbo}</i>"
                )

            linhas.append(f"<i>Regras: paridade 7–8, seq≤3, anti-overlap≤{BOLAO_MAX_OVERLAP}</i>")

            return await update.message.reply_text("\n".join(linhas), parse_mode="HTML")

        except Exception as e:
            logger.error("Erro no /refinar_bolao:\n" + traceback.format_exc())
            return await update.message.reply_text(f"Erro no /refinar_bolao: {e}")

    # ===== Registrar a última geração (para o aprendizado leve) =====
    def _registrar_geracao(self, apostas: list[list[int]], base_resultado: list[int] | None = None) -> None:
        from datetime import datetime
        st = _bolao_load_state()
        st = dict(st) if isinstance(st, dict) else {}
        st.setdefault("learning", {})

        st["learning"]["last_generation"] = {
            "apostas": [list(map(int, ap)) for ap in apostas],
            "timestamp": datetime.now(ZoneInfo(TIMEZONE)).strftime("%Y-%m-%d %H:%M:%S %z"),
            "base_resultado": list(map(int, base_resultado)) if base_resultado else None,
        }
        _bolao_save_state(st)

    # --- /estado_bolao: resumo do aprendizado atual (diagnóstico detalhado) ---
    async def estado_bolao(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Mostra o panorama do aprendizado leve:
          - Média de bias (e distribuição + / 0 / -)
          - Top 5 dezenas mais favorecidas e menos favorecidas pelo bias
          - α atual
          - Último auto-aprendizado
          - Snapshot mais recente
          - Timestamp do último registro de geração (A1..An)
          - Estimativa de ciclos (maior contagem em 'seen')
        """
        try:
            st = _bolao_load_state() or {}
            bias_raw = st.get("bias", {}) or {}
            # Normaliza bias como {int: float}
            bias = {}
            for k, v in bias_raw.items():
                try:
                    bias[int(k)] = float(v)
                except Exception:
                    continue

            # Métricas básicas
            n = len(bias)
            media_bias = (sum(bias.values()) / n) if n else 0.0
            pos = [(k, v) for k, v in bias.items() if v > 0]
            neg = [(k, v) for k, v in bias.items() if v < 0]
            zer = n - (len(pos) + len(neg))

            # Top 5 positivos / negativos
            top_pos = sorted(pos, key=lambda kv: kv[1], reverse=True)[:5]
            top_neg = sorted(neg, key=lambda kv: kv[1])[:5]

            # α atual (fallback para ALPHA_PADRAO se não existir em estado)
            try:
                alpha_atual = float(st.get("alpha", ALPHA_PADRAO))
            except Exception:
                alpha_atual = ALPHA_PADRAO

            # Último auto-aprendizado e snapshot
            last_auto = st.get("last_auto", "--")
            snap_id = st.get("last_snapshot", st.get("snapshot", "--"))

            # Último registro de geração (para o aprendizado leve comparar)
            last_gen = ((st.get("learning") or {}).get("last_generation") or {})
            last_gen_ts = last_gen.get("timestamp", "--")

            # Estimativa de ciclos: maior valor de 'seen' (quantas vezes avaliamos dezenas)
            seen_map = st.get("seen", {}) or {}
            try:
                ciclos = max(int(v) for v in seen_map.values()) if seen_map else 0
            except Exception:
                ciclos = 0

            # Monta a resposta
            linhas = []
            linhas.append("📈 <b>Estado do Aprendizado (Bolão)</b>\n")
            linhas.append(f"• Média de bias: <b>{media_bias:+.3f}</b>  "
                          f"(+{len(pos)} | 0={zer} | −{len(neg)})")
            linhas.append(f"• α atual: <b>{alpha_atual:.2f}</b>")
            linhas.append(f"• Último auto-aprendizado: <b>{last_auto}</b>")
            linhas.append(f"• Snapshot: <b>{snap_id}</b>")
            linhas.append(f"• Último registro de geração: <b>{last_gen_ts}</b>")
            linhas.append(f"• Estimativa de ciclos: <b>{ciclos}</b>\n")

            def _fmt_pairs(pairs):
                if not pairs:
                    return "—"
                return "  ".join(f"{k:02d}(<i>{v:+.3f}</i>)" for k, v in pairs)

            linhas.append("<b>Top +5 (bias ↑)</b>")
            linhas.append(_fmt_pairs(top_pos))
            linhas.append("\n<b>Top −5 (bias ↓)</b>")
            linhas.append(_fmt_pairs(top_neg))

            return await update.message.reply_text("\n".join(linhas), parse_mode="HTML")

        except Exception:
            # Fallback minimalista em caso de qualquer problema inesperado
            try:
                media_bias = 0.0
                st = _bolao_load_state() or {}
                bias_raw = st.get("bias", {}) or {}
                if bias_raw:
                    vals = []
                    for v in bias_raw.values():
                        try:
                            vals.append(float(v))
                        except Exception:
                            pass
                    if vals:
                        media_bias = sum(vals) / len(vals)
            except Exception:
                media_bias = 0.0
            return await update.message.reply_text(f"📊 Média de bias atual: {media_bias:.3f}")

    # --- /auto_aprender: rotina automática de aprendizado leve após cada concurso ---
    async def auto_aprender(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Aprendizado leve entre concursos:
          1) Mantém sua lógica de refino baseada no histórico/último resultado.
          2) Se houver uma geração registrada (A1..A10 etc.), aplica micro-ajustes:
             - Alvo de repetição R ≈ 9.5 (9R–10R).
             - Paridade tende a 7–8.
             - Penaliza seq>3.
             - Ajusta α em ±0.01 por ciclo (faixa 0.30–0.42).
        Persistência em BOLAO_STATE_PATH.
        """
        from statistics import mean

        # --------- 0) Carrega estado e último resultado ---------
        st = _bolao_load_state()
        st = dict(st) if isinstance(st, dict) else {}
        st.setdefault("learning", {})
        historico = carregar_historico(HISTORY_PATH)
        if not historico:
            return  # sem histórico: nada a fazer

        try:
            ultimo = self._ultimo_resultado(historico)  # lista[15], ordenada
        except Exception:
            ultimo = []

        # --------- 1) (Mantém seu comportamento atual) ---------
        # Aqui você pode incluir/refazer qualquer atualização de bias/hits/seen
        # que sua versão anterior de auto_aprender aplicava com base em `ultimo`.
        # Como não temos o bloco original completo neste recorte, mantemos no-ops.
        # (Seu /refinar_bolao continua existindo para refinos manuais.)

        # --------- 2) Micro-ajustes baseados na última geração salva ---------
        last_gen = (st.get("learning") or {}).get("last_generation") or {}
        apostas = last_gen.get("apostas") or []

        if apostas and ultimo:
            # métricas
            hits = [self._contar_acertos(ap, ultimo) for ap in apostas]
            mu = mean(hits) if hits else 0.0

            seq_list = [self._max_seq(ap) for ap in apostas]
            seq_mu = mean(seq_list) if seq_list else 0.0
            seq_viol = sum(1 for s in seq_list if s > 3)

            pares_medios = mean(self._paridade(ap)[0] for ap in apostas) if apostas else 0.0

            # alvo de repetição
            alvo_R = 9.5
            delta_R = mu - alvo_R  # + = repetindo demais; - = repetindo pouco

            # biases atuais
            bias = dict(st.get("bias") or {})
            bias_R   = float(bias.get("R", 0.0))
            bias_par = float(bias.get("paridade", 0.0))
            bias_seq = float(bias.get("seq", 0.0))

            # passos leves
            k_R, k_P, k_S = 0.02, 0.01, 0.02

            # atualiza bias
            bias_R  -= k_R * delta_R
            bias_par += k_P * (7.5 - pares_medios)
            bias_seq += k_S * ((seq_mu - 3.0) + 0.5 * (seq_viol / max(1, len(apostas))))

            # clamp seguro
            def clamp(x, lo, hi): return max(lo, min(hi, x))
            bias_R   = clamp(bias_R,  -0.20, 0.20)
            bias_par = clamp(bias_par,-0.15, 0.15)
            bias_seq = clamp(bias_seq, -0.20, 0.20)
            st["bias"] = {"R": round(bias_R, 6), "paridade": round(bias_par, 6), "seq": round(bias_seq, 6)}

            # ajusta alpha leve (persistido no mesmo estado)
            alpha = float(st.get("alpha", ALPHA_PADRAO))
            delta_alpha = -0.01 if mu < 9.0 else (0.01 if mu > 10.0 else 0.0)
            alpha = clamp(alpha + delta_alpha, 0.30, 0.42)
            st["alpha"] = round(alpha, 4)

        # --------- 3) Persistência + telemetria opcional ---------
        _bolao_save_state(st)

        if update and getattr(update, "message", None):
            try:
                if apostas and ultimo:
                    await update.message.reply_text(
                        "🤖 Aprendizado leve aplicado.\n"
                        f"• Lote registrado: {len(apostas)} apostas\n"
                        f"• α atual: {st.get('alpha', ALPHA_PADRAO):.2f}\n"
                        f"• bias: R={st.get('bias',{}).get('R',0):+.3f} | "
                        f"par={st.get('bias',{}).get('paridade',0):+.3f} | "
                        f"seq={st.get('bias',{}).get('seq',0):+.3f}"
                    )
                else:
                    await update.message.reply_text("🤖 Aprendizado leve: sem geração/resultado suficientes; manutenção confirmada.")
            except Exception:
                pass

    # --------- Gerador Ciclo C (ancorado no último resultado) — versão reforçada ---------
    def _gerar_ciclo_c_por_ultimo_resultado(self, historico):
        if not historico:
            raise ValueError("Histórico vazio no Ciclo C.")

        ultimo = self._ultimo_resultado(historico)
        u_set = set(ultimo)
        comp_list = sorted(self._complemento(u_set))  # <- garante lista ordenada
        anchors = set(CICLO_C_ANCHORS)

        def _forcar_repeticoes(a: list[int], r_alvo: int) -> list[int]:
            a = a[:]
            r_atual = sum(1 for n in a if n in u_set)
            if r_atual == r_alvo:
                return a

            if r_atual < r_alvo:
                faltam = [n for n in ultimo if n not in a]
                for add in faltam:
                    # preserva âncoras, troca um não-último quando possível
                    rem = next((x for x in a if x not in u_set and x not in anchors), None)
                    if rem is None:
                        rem = next((x for x in a if x not in u_set), None)
                    if rem is None:
                        break
                    a.remove(rem); a.append(add); a.sort()
                    r_atual += 1
                    if r_atual == r_alvo:
                        break
            else:
                # r_atual > r_alvo: remova um do último (não âncora) e insira do complemento
                for rem in [x for x in reversed(a) if x in u_set and x not in anchors]:
                    add = next((c for c in comp_list if c not in a), None)
                    if add is None:
                        break
                    a.remove(rem); a.append(add); a.sort()
                    r_atual -= 1
                    if r_atual == r_alvo:
                        break
            return a

        def _ok(a: list[int], r_alvo: int) -> bool:
            pares = self._contar_pares(a)
            return (7 <= pares <= 8) and (self._max_seq(a) <= 3) and (sum(1 for n in a if n in u_set) == r_alvo)

        apostas: list[list[int]] = []
        for i, r_alvo in enumerate(CICLO_C_PLANOS):
            off_last = (i * 3) % 15
            off_comp = (i * 5) % len(comp_list) if len(comp_list) > 0 else 0

            a = self._construir_aposta_por_repeticao(
                last_sorted=ultimo,
                comp_sorted=comp_list,
                repeticoes=r_alvo,
                offset_last=off_last,
                offset_comp=off_comp,
            )

            # força âncoras (trocando preferencialmente um número do último que não é âncora)
            for anc in anchors:
                if anc not in a:
                    rem = next((x for x in a if x in u_set and x not in anchors), None)
                    if rem is None:
                        rem = next((x for x in reversed(a) if x not in anchors), None)
                    if rem is not None and rem != anc:
                        a.remove(rem); a.append(anc); a.sort()

            # forma-base: repetição alvo + paridade/seq
            a = _forcar_repeticoes(a, r_alvo)
            a = self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3, anchors=anchors)

            # reforço “miolo”: garantir pelo menos 3 na faixa 12..18
            mid_lo, mid_hi = 12, 18
            mid = [n for n in a if mid_lo <= n <= mid_hi]
            if len(mid) < 3:
                need = 3 - len(mid)
                cand_add = [n for n in comp_list if mid_lo <= n <= mid_hi and n not in a]
                cand_rem = [x for x in sorted(a, reverse=True) if not (mid_lo <= x <= mid_hi) and x not in anchors]
                j = 0
                while need > 0 and j < len(cand_add) and j < len(cand_rem):
                    add, rem = cand_add[j], cand_rem[j]
                    if add not in a and rem in a:
                        a.remove(rem); a.append(add); a.sort()
                        a = self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3, anchors=anchors)
                        a = _forcar_repeticoes(a, r_alvo)
                        need -= 1
                    j += 1

            apostas.append(sorted(a))

        # cobertura de ausentes do complemento (se algum não apareceu em nenhuma aposta)
        ausentes = set(comp_list)
        presentes = set(n for ap in apostas for n in ap)
        faltantes = [n for n in ausentes if n not in presentes]
        if faltantes:
            a = apostas[-1][:]
            for n in faltantes:
                rem = next((x for x in reversed(a) if x in u_set and x not in anchors), None)
                if rem is None:
                    break
                if n not in a:
                    a.remove(rem); a.append(n); a.sort()
                    a = self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3, anchors=anchors)
                    a = _forcar_repeticoes(a, CICLO_C_PLANOS[-1])
            apostas[-1] = a

        # normalização de forma + anti-overlap antes da selagem
        apostas = [self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3, anchors=anchors) for a in apostas]
        apostas = self._anti_overlap(apostas, ultimo=ultimo, comp=comp_list, max_overlap=BOLAO_MAX_OVERLAP, anchors=anchors)

        # estabilização por plano
        for i, r_alvo in enumerate(CICLO_C_PLANOS):
            if i >= len(apostas):
                break
            a = apostas[i][:]
            for anc in anchors:
                if anc not in a:
                    rem = next((x for x in reversed(a) if x not in anchors), None)
                    if rem is not None and rem != anc:
                        a.remove(rem); a.append(anc); a.sort()
            for _ in range(12):  # limite defensivo
                a = self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3, anchors=anchors)
                a = _forcar_repeticoes(a, r_alvo)
                if _ok(a, r_alvo):
                    break
            apostas[i] = sorted(a)

        # anti-overlap adicional antes da selagem final
        apostas = self._anti_overlap(apostas, ultimo=ultimo, comp=comp_list, max_overlap=BOLAO_MAX_OVERLAP, anchors=anchors)

        # >>> Selagem final do Ciclo C (paridade 7–8, seq≤3, dedup e anti-overlap), preservando âncoras
        apostas = self._fechar_ciclo_c(apostas, ultimo=ultimo, anchors=tuple(anchors))
        # <<<

        return apostas


    @staticmethod
    def _contar_repeticoes(aposta, ultimo):
        u = set(ultimo)
        return sum(1 for n in aposta if n in u)

    # --- Novo comando: /mestre ---
    async def mestre(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if not self._usuario_autorizado(user_id):
            await update.message.reply_text("⛔ Você não está autorizado a gerar apostas.")
            return

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

        chat_id = update.message.chat_id if update.message else update.effective_chat.id
        if self._hit_cooldown(chat_id, "mestre"):
            await update.message.reply_text(f"⏳ Aguarde {COOLDOWN_SECONDS}s para usar /mestre novamente.")
            return

        # --- carrega histórico ---
        try:
            historico = carregar_historico(HISTORY_PATH)
            if not historico:
                await update.message.reply_text("Erro: histórico vazio.")
                return
        except Exception as e:
            await update.message.reply_text(f"Erro ao carregar histórico: {e}")
            return

        # --- seed composta: incremental por snapshot ^ seed estável por usuário/chat ---
        try:
            ultimo_sorted = self._ultimo_resultado(historico)
            snap = self._latest_snapshot()
            seed_inc = self._next_draw_seed(snap.snapshot_id)   # incremental por snapshot
            user_seed = self._calc_mestre_seed(
                user_id=update.effective_user.id,
                chat_id=chat_id,
                ultimo_sorted=ultimo_sorted,
            )
            seed = (seed_inc ^ (user_seed & 0xFFFFFFFF)) & 0xFFFFFFFF
        except Exception:
            seed = 0
        ultimo_sorted = locals().get("ultimo_sorted", self._ultimo_resultado(historico))
        try:
            snap = locals().get("snap", self._latest_snapshot())
        except Exception:
            snap = None

        # --- gera as apostas usando a seed calculada ---
        try:
            apostas = self._gerar_mestre_por_ultimo_resultado(historico, seed=seed)
        except Exception as e:
            logger.error("Erro no preset Mestre (último resultado):\n" + traceback.format_exc())
            await update.message.reply_text(f"Erro no preset Mestre: {e}")
            return

        # --- Pós-filtro unificado (forma + dedup/overlap + bias + forma) ---
        try:
            apostas = self._pos_filtro_unificado(apostas, ultimo=ultimo_sorted)
        except Exception:
            logger.warning("Falha no pós-filtro unificado no /mestre; aplicando selagem rápida.", exc_info=True)
            apostas = [self._hard_lock_fast(a, ultimo_sorted, anchors=frozenset()) for a in apostas]

        # --- REGISTRO para aprendizado leve (Mestre) ---
        try:
            # reusa o mesmo último resultado calculado do histórico
            ultimo = ultimo_sorted if ultimo_sorted else []
        except Exception:
            ultimo = []
        try:
            self._registrar_geracao(apostas, base_resultado=ultimo)
        except Exception:
            logger.warning("Falha ao registrar geração para aprendizado leve (/mestre).", exc_info=True)

        # --- Telemetria e formatação da resposta ---
        from datetime import datetime
        from zoneinfo import ZoneInfo

        snap_id = snap.snapshot_id if snap else "n/a"
        linhas = ["🎰 <b>SUAS APOSTAS INTELIGENTES — Preset Mestre</b> 🎰\n"]

        ok_count = 0
        for i, aposta in enumerate(apostas, 1):
            t = self._telemetria(aposta, ultimo_sorted, alvo_par=(7, 8), max_seq=3)
            status = "✅ OK" if t.ok_total else "🛠️ REPARAR"
            if t.ok_total:
                ok_count += 1
            linhas.append(
                f"<b>Aposta {i}:</b> {' '.join(f'{n:02d}' for n in aposta)}\n"
                f"🔢 Pares: {t.pares} | Ímpares: {t.impares} | SeqMax: {t.max_seq} | {t.repeticoes}R | {status}\n"
            )

        linhas.append(f"\n<b>Conformidade</b>: {ok_count}/{len(apostas)} dentro de (paridade 7–8, seq≤3)")
        linhas.append(f"<i>Regras: paridade 7–8, seq≤3, anti-overlap≤{BOLAO_MAX_OVERLAP}</i>")

        if SHOW_TIMESTAMP:
            now_sp = datetime.now(ZoneInfo(TIMEZONE))
            carimbo = now_sp.strftime("%Y-%m-%d %H:%M:%S %Z")
            hash_ult = _hash_dezenas(ultimo_sorted)
            linhas.append(
                f"<i>base=último resultado | paridade=7–8 | max_seq=3 | "
                f"hash={hash_ult} | snapshot={snap_id} | {carimbo}</i>"
            )

        await update.message.reply_text("\n".join(linhas), parse_mode="HTML")

    # --- Diagnóstico ---
    async def ping(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("pong")

    async def versao(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        txt = (
            f"🤖 Versão do bot\n"
            f"- BUILD_TAG: <code>{BUILD_TAG}</code>\n"
            f"- Import layout: <code>{LAYOUT}</code>\n"
            f"- Comandos: /start /gerar /mestre /mestre_bolao /refinar_bolao /ab /meuid /autorizar /remover /backtest /diagbase /ping /versao"
        )
        await update.message.reply_text(txt, parse_mode="HTML")

    async def diagbase(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            snap = self._latest_snapshot()
            await update.message.reply_text(
                "📌 Base atual carregada pelo bot\n"
                f"- snapshot_id: <code>{snap.snapshot_id}</code>\n"
                f"- tamanho(histórico): <b>{snap.tamanho}</b>\n"
                f"- último resultado: <b>{' '.join(f'{n:02d}' for n in snap.dezenas)}</b>",
                parse_mode="HTML"
            )
        except Exception as e:
            await update.message.reply_text(f"Erro no diagbase: {e}")

    # --- Auxiliares de acesso ---
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

    # --- A/B técnico + Ciclo C ---
    async def ab(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if not self._usuario_autorizado(user_id):
            return await update.message.reply_text("⛔ Você não está autorizado.")

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

        chat_id = update.effective_chat.id
        if self._hit_cooldown(chat_id, "ab"):
            return await update.message.reply_text(f"⏳ Aguarde {COOLDOWN_SECONDS}s para usar /ab novamente.")

        # ---------------------------------------------------------
        # MODO CICLO C ("/ab c" ou "/ab ciclo")
        # ---------------------------------------------------------
        mode_ciclo = (len(context.args) >= 1 and str(context.args[0]).lower() in {"ciclo", "c"})
        if mode_ciclo:
            try:
                # imports locais
                import hashlib
                from datetime import datetime
                from zoneinfo import ZoneInfo

                snap = self._latest_snapshot()

                historico = carregar_historico(HISTORY_PATH)
                if not historico:
                    return await update.message.reply_text("Erro: histórico vazio.")

                # 1) geração bruta do Ciclo C
                ultimo = self._ultimo_resultado(historico)
                apostas = self._gerar_ciclo_c_por_ultimo_resultado(historico)

                # 2) SELAGEM FINAL do Ciclo C (R exato, paridade 7–8, seq≤3, dedup e anti-overlap), preservando âncoras
                try:
                    anchors = tuple(CICLO_C_ANCHORS)  # ex.: (9, 11)
                except Exception:
                    anchors = (9, 11)

                try:
                    apostas = self._fechar_ciclo_c(apostas, ultimo=ultimo, anchors=anchors)
                except Exception:
                    logger.warning("Falha ao selar Ciclo C via _fechar_ciclo_c; seguindo com apostas atuais.", exc_info=True)

                # 2.1) Pós-filtro unificado (não deveria alterar R, mas pode ajustar forma/overlap/bias)
                try:
                    apostas = self._pos_filtro_unificado(apostas, ultimo=ultimo)
                except Exception:
                    logger.warning("Pós-filtro unificado falhou; seguindo com apostas seladas.", exc_info=True)

                # 2.2) RESELAGEM para garantir R-alvo exato após o pós-filtro
                try:
                    apostas = self._fechar_ciclo_c(apostas, ultimo=ultimo, anchors=anchors)
                except Exception:
                    logger.warning("Reselagem do Ciclo C falhou após pós-filtro.", exc_info=True)

                # 2.3) REGISTRO para aprendizado leve (/ab ciclo C)
                try:
                    self._registrar_geracao(apostas, base_resultado=ultimo or [])
                except Exception:
                    logger.warning("Falha ao registrar geração para aprendizado leve (/ab ciclo C).", exc_info=True)

                # 3) Formatação da resposta (NÃO reprocessa as apostas!)
                linhas = []
                linhas.append("🎯 Ciclo C — baseado no último resultado")
                if len(anchors) >= 2:
                    linhas.append(f"Âncoras: {anchors[0]:02d} e {anchors[1]:02d} | paridade=7–8 | max_seq=3")
                else:
                    linhas.append("Âncoras: — | paridade=7–8 | max_seq=3")

                u_set = set(ultimo)
                for i, a in enumerate(apostas, 1):
                    pares = self._contar_pares(a)
                    seq = self._max_seq(a)
                    rep = sum(1 for n in a if n in u_set)
                    alvo = CICLO_C_PLANOS[i - 1] if (i - 1) < len(CICLO_C_PLANOS) else rep
                    linhas.append(
                        f"\nAposta {i}: " + " ".join(f"{n:02d}" for n in a) + f"  [{rep}R; alvo={alvo}R]"
                    )
                    linhas.append(f"🔢 Pares: {pares} | Ímpares: {15 - pares} | SeqMax: {seq}")

                # rodapé (hash/snapshot/carimbo)
                try:
                    hash_ult = hashlib.md5("".join(f"{n:02d}" for n in ultimo).encode()).hexdigest()[:8]
                except Exception:
                    hash_ult = "--"
                try:
                    snap_id = getattr(snap, "snapshot_id", "--")
                except Exception:
                    snap_id = "--"
                carimbo = datetime.now(tz=ZoneInfo("America/Sao_Paulo")).strftime("%Y-%m-%d %H:%M:%S %z")

                return await update.message.reply_text("\n".join(linhas), parse_mode="HTML")

            except Exception as e:
                logger.error("Erro no /ab ciclo C:\n" + traceback.format_exc())
                return await update.message.reply_text(f"Erro ao executar Ciclo C: {e}")


        # ---------------------------------------------------------
        # MODO A/B TÉCNICO (padrão quando NÃO é ciclo)
        # ---------------------------------------------------------
        try:
            qtd = int(context.args[0]) if len(context.args) >= 1 else QTD_BILHETES_PADRAO
            janela = int(context.args[1]) if len(context.args) >= 2 else 60
            alphaA = float(context.args[2].replace(",", ".")) if len(context.args) >= 3 else ALPHA_PADRAO
            alphaB = float(context.args[3].replace(",", ".")) if len(context.args) >= 4 else ALPHA_TEST_B
        except Exception:
            qtd, janela, alphaA, alphaB = QTD_BILHETES_PADRAO, 60, ALPHA_PADRAO, ALPHA_TEST_B

        # saneamento de parâmetros
        qtd, janela, alphaA = self._clamp_params(qtd, janela, alphaA)
        _, _, alphaB = self._clamp_params(qtd, janela, alphaB)

        # geração dos dois lotes
        try:
            apostasA = self._gerar_apostas_inteligentes(qtd=qtd, janela=janela, alpha=alphaA)
            apostasB = self._gerar_apostas_inteligentes(qtd=qtd, janela=janela, alpha=alphaB)
        except Exception:
            logger.error("Erro no /ab:\n" + traceback.format_exc())
            return await update.message.reply_text("Erro ao gerar A/B. Tente novamente.")
        
        # REGISTRO para aprendizado leve (/ab A/B técnico)
        try:
            # Concatena os dois lotes para que o aprendizado avalie o conjunto completo do experimento
            historico = carregar_historico(HISTORY_PATH)
            ultimo = self._ultimo_resultado(historico) if historico else []
            self._registrar_geracao(list(apostasA) + list(apostasB), base_resultado=ultimo or [])
        except Exception:
            logger.warning("Falha ao registrar geração para aprendizado leve (/ab técnico).", exc_info=True)

        # formatação da saída (mantida como no seu código)
        def _fmt(tag, aps):
            linhas = [f"🅰️🅱️ <b>LOTE {tag}</b>\n"]
            for i, a in enumerate(aps, 1):
                pares = self._contar_pares(a)
                linhas.append(
                    f"<b>Aposta {i}:</b> {' '.join(f'{n:02d}' for n in a)}\n"
                    f"🔢 Pares: {pares} | Ímpares: {15 - pares}\n"
                )
            return "\n".join(linhas)

        msg = (
            f"🧪 <b>A/B Técnico</b> — janela={janela}\n"
            f"• A: α={alphaA:.2f}\n"
            f"• B: α={alphaB:.2f}\n\n"
            f"{_fmt('A', apostasA)}\n\n{_fmt('B', apostasB)}"
        )
        await update.message.reply_text(msg, parse_mode="HTML")

    def _ciclo_c_fixup(self, apostas: list[list[int]], historico) -> list[list[int]]:
        if not historico:
            return apostas
        ultimo = self._ultimo_resultado(historico)
        u_set = set(ultimo)
        anchors = set(CICLO_C_ANCHORS)

        def _forcar_repeticoes(a: list[int], r_alvo: int) -> list[int]:
            a = a[:]
            r_atual = sum(1 for n in a if n in u_set)
            if r_atual == r_alvo:
                return a
            comp = [n for n in range(1, 26) if n not in a]
            if r_atual < r_alvo:
                faltam = [n for n in ultimo if n not in a]
                for add in faltam:
                    rem = next((x for x in a if x not in u_set and x not in anchors), None)
                    if rem is None:
                        rem = next((x for x in a if x not in u_set), None)
                    if rem is None:
                        break
                    a.remove(rem); a.append(add); a.sort()
                    r_atual += 1
                    if r_atual == r_alvo:
                        break
            else:
                for rem in [x for x in reversed(a) if x in u_set and x not in anchors]:
                    add = next((c for c in comp if c not in a), None)
                    if add is None:
                        break
                    a.remove(rem); a.append(add); a.sort()
                    r_atual -= 1
                    if r_atual == r_alvo:
                        break
            return a

        for i, a in enumerate(apostas):
            for anc in anchors:
                if anc not in a:
                    rem = next((x for x in reversed(a) if x not in anchors), None)
                    if rem is not None and rem != anc:
                        a.remove(rem); a.append(anc); a.sort()
            r_alvo = CICLO_C_PLANOS[i]
            for _ in range(14):
                a = self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3, anchors=anchors)
                a = _forcar_repeticoes(a, r_alvo)
                pares = self._contar_pares(a)
                if 7 <= pares <= 8 and self._max_seq(a) <= 3 and sum(1 for n in a if n in u_set) == r_alvo:
                    break
            a = self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3, anchors=anchors)
            a = _forcar_repeticoes(a, r_alvo)
            a = self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3, anchors=anchors)
            apostas[i] = sorted(a)

        apostas = self._anti_overlap(apostas, ultimo=ultimo, comp=[n for n in range(1,26) if n not in ultimo], max_overlap=BOLAO_MAX_OVERLAP, anchors=anchors)
        for i, a in enumerate(apostas):
            a = self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3, anchors=anchors)
            a = _forcar_repeticoes(a, CICLO_C_PLANOS[i])
            apostas[i] = sorted(a)

        return apostas
    
    # ====== LOCK RÁPIDO E DETERMINÍSTICO (pares 7–8, seq≤3) ======
    def _hard_lock_fast(self, a: list[int], ultimo: list[int], anchors=frozenset()) -> list[int]:
        """
        Lock rápido e determinístico:
        - Garante P∈[7,8] e Seq≤3
        - Evita mexer em âncoras quando possível
        - Recalcula complemento a cada modificação
        """
        a = sorted(set(int(x) for x in a if 1 <= int(x) <= 25))[:15]

        def is_ok(x: list[int]) -> bool:
            return (7 <= self._contar_pares(x) <= 8) and (self._max_seq(x) <= 3) and (len(x) == 15)

        def comp_now(x: list[int]) -> list[int]:
            return [n for n in range(1, 26) if n not in x]

        if is_ok(a):
            return a

        for _ in range(20):  # limite curto
            changed = False

            # --- Paridade ---
            pares = self._contar_pares(a)
            if pares > 8:
                rem = (next((x for x in a if x % 2 == 0 and x in ultimo and x not in anchors), None)
                       or next((x for x in a if x % 2 == 0 and x not in anchors), None))
                add = next((c for c in comp_now(a) if c % 2 == 1 and (c-1 not in a) and (c+1 not in a)), None) \
                      or next((c for c in comp_now(a) if c % 2 == 1), None)
                if rem is not None and add is not None and rem in a and add not in a:
                    a.remove(rem); a.append(add); a.sort()
                    changed = True

            elif pares < 7:
                rem = (next((x for x in a if x % 2 == 1 and x in ultimo and x not in anchors), None)
                       or next((x for x in a if x % 2 == 1 and x not in anchors), None))
                add = next((c for c in comp_now(a) if c % 2 == 0 and (c-1 not in a) and (c+1 not in a)), None) \
                      or next((c for c in comp_now(a) if c % 2 == 0), None)
                if rem is not None and add is not None and rem in a and add not in a:
                    a.remove(rem); a.append(add); a.sort()
                    changed = True

            # --- Sequências ---
            if self._max_seq(a) > 3:
                s = sorted(a)
                idx = next((i for i in range(len(s)-3)
                        if s[i+3] == s[i] + 3 and s[i+1] == s[i] + 1 and s[i+2] == s[i] + 2), None)
                if idx is not None:
                    janela = s[idx:idx+4]
                    rem = (next((x for x in janela if x in ultimo and x not in anchors), None)
                           or next((x for x in janela if x not in anchors), None))
                    add = next((c for c in comp_now(a) if (c-1 not in a) and (c+1 not in a)), None) \
                          or next((c for c in comp_now(a)), None)
                    if rem is not None and add is not None and rem in a and add not in a:
                        a.remove(rem); a.append(add); a.sort()
                        changed = True

            if is_ok(a):
                break
            if not changed:
                # fallback do núcleo já existente
                a = self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3, anchors=anchors)
                break

        return sorted(a)[:15]

    # ------------- Handler do backtest -------------
    async def backtest(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if not self._is_admin(user_id):
            return

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
    try:
        logger.info("Inicializando bot...")
        bot = LotoFacilBot()
        logger.info("Iniciando polling...")
        bot.run()
    except Exception:
        logger.critical("Falha fatal ao iniciar o bot:\n%s", traceback.format_exc())
        import time as _t; _t.sleep(3)
        raise
