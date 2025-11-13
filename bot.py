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
from collections import Counter, defaultdict
from itertools import combinations
from functools import partial
from typing import List, Set, Tuple, Optional
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

# ==== Política de qualidade de lote ====
DESEMPENHO_MINIMO_R = 11.0     # média mínima aceitável
DESEMPENHO_BOM_R    = 12.0     # opcional: faixa "bom"

# ========================
# Parâmetros padrão do gerador
# ========================
# Quantidade: permitir até 50 no /gerar
QTD_BILHETES_PADRAO = 5
QTD_BILHETES_MIN = 1
QTD_BILHETES_MAX = 200

SHOW_TIMESTAMP = True
TIMEZONE = "America/Sao_Paulo"

# Janela e alpha (alinhados ao utils/backtest defaults/amarras)
JANELA_PADRAO = 60
JANELA_MIN, JANELA_MAX = 50, 1000

ALPHA_PADRAO = 0.36
ALPHA_MIN,  ALPHA_MAX  = 0.05, 0.95

HISTORY_PATH = "data/history.csv"
WHITELIST_PATH = "whitelist.txt"

# --- Alpha lock (apenas no /gerar) ---
LOCK_ALPHA_GERAR = True      # deixe True para travar /gerar em 0.36
ALPHA_LOCK_VALUE  = 0.36     # valor travado só para /gerar

# Cooldown (segundos) para evitar flood
COOLDOWN_SECONDS = 10

# Limite seguro para mensagens no Telegram
TELEGRAM_SAFE_MAX = 3900  # 4096 é o real; deixamos folga por causa do HTML

# Identificação do build (para /versao)
BUILD_TAG = getenv("BUILD_TAG", "unknown")

# ========================
# Configurações do Bolão Inteligente v5 (19 → 15)
# ========================
BOLAO_JANELA = 60
BOLAO_ALPHA  = 0.36
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
# Bolão Matriz 20 → 15 (A/B/C) + Estado Persistente
# ========================
from pathlib import Path

# Diretório base de dados (mesmo local onde fica o history.csv)
DATA_DIR = str(Path(HISTORY_PATH).parent)
os.makedirs(DATA_DIR, exist_ok=True)

BOLAO20_JANELA = 60
BOLAO20_ALPHA  = 0.30
BOLAO20_PAR    = (7, 8)
BOLAO20_MAXSEQ = 3
BOLAO20_MAX_OVERLAP = 11

BOLAO20_PLANOS = {"A": 10, "B": 20, "C": 48}
BOLAO20_DESCR  = {
    "A": "10 jogos — 14 frequente (não garantido)",
    "B": "20 jogos — 14 muito frequente (estilo lotérica)",
    "C": "36–48 jogos — garantia técnica de 14 se errar ≤1"
}

# Arquivo de estado persistente (para conferência / auditoria do bolão)
BOLAO20_STATE_PATH = os.path.join(DATA_DIR, "bolao20_state.json")
BOLAO20_AUTO_MATCH_LAST = True  # tenta casar sessão com último oficial automaticamente

# ========================
# Aprendizado REAL — paths e grades
# ========================
LEARN_LOG_PATH = "data/learn_log.csv"   # log de (snapshot_id, oficial, apostas, placares)
REAL_STATE_PATH = BOLAO_STATE_PATH      # reaproveita o mesmo arquivo de estado

# Política de aprendizado (gating por oficial)
LEARN_POLICY = "official_gate"   # ["official_gate", "free_run"]
IMPROVE_EPS  = 0.15              # melhoria mínima de score TOP-K para aceitar novo α/janela
BIAS_LR      = 0.10              # passo do ajuste suave de bias quando aplicar

# Grade de busca (pode ajustar depois)
ALPHA_GRID = [round(x, 2) for x in (0.28, 0.30, 0.31, 0.33, 0.36, 0.39, 0.40, 0.42)]
JANELA_GRID = [50, 60, 80, 100]
GRID_MAX_TIME_S = 4.0  # limite duro pra não travar bot

# Tamanho mínimo de histórico para aprendizado real
MIN_TREINO = 120  # concursos
ROLLING_TEST = 12 # últimos T pontos para validação walk-forward
TOPK_SCORE = 5    # média dos TOP-K bilhetes por concurso na métrica de acertos

def _score_lote(apostas: list[list[int]], oficial: list[int]) -> list[int]:
    """Conta acertos por aposta (15 dezenas)."""
    o = set(oficial)
    return [sum(1 for n in a if n in o) for a in apostas]

def _hits_media_topk(placares: list[int], k: int = TOPK_SCORE) -> float:
    if not placares:
        return 0.0
    k = max(1, min(k, len(placares)))
    return sum(sorted(placares, reverse=True)[:k]) / float(k)

def _append_learn_log(snapshot_id: str, oficial: list[int], apostas: list[list[int]]):
    """Grava uma linha no CSV de aprendizado real (sem travar se falhar)."""
    try:
        import csv, os
        os.makedirs(os.path.dirname(LEARN_LOG_PATH), exist_ok=True)
        placar = _score_lote(apostas, oficial) if oficial else []
        with open(LEARN_LOG_PATH, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                snapshot_id,
                " ".join(f"{n:02d}" for n in sorted(oficial)) if oficial else "",
                "|".join(" ".join(f"{x:02d}" for x in a) for a in apostas),
                " ".join(str(p) for p in placar),
                datetime.now(ZoneInfo(TIMEZONE)).strftime("%Y-%m-%d %H:%M:%S")
            ])
    except Exception:
        logger.warning("Falha ao registrar linha no learn_log.csv", exc_info=True)

# ========================
# Heurísticas adicionais (Mestre + A/B)
# ========================
# Pares cuja coocorrência derrubou média em análises anteriores
PARES_PENALIZADOS = {(23, 2), (22, 19), (24, 20), (11, 1)}
# Conjunto de "ruídos" com cap de frequência por lote (Mestre)
RUIDOS = {2, 1, 14, 19, 20, 10, 7, 15, 21, 9}
# No pacote de 10 apostas do Mestre, cada ruído pode aparecer no máx. 6 apostas
RUIDO_CAP_POR_LOTE = 6

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

def _normalize_state_defaults(state: dict) -> dict:
    state = dict(state or {})
    # política de aprendizado: “gated” por oficial, a menos que você troque por “free_run”
    state.setdefault("learn_policy", LEARN_POLICY)
    # fila de lotes pendentes de avaliação (cada /gerar, /mestre, etc, empilha aqui — sem aprender ainda)
    state.setdefault("pending_batches", [])
    # baseline (score/α/janela) usado para decidir se um novo parâmetro realmente melhorou
    state.setdefault(
        "last_baseline",
        {
            "score": 0.0,
            "alpha": state.get("alpha", ALPHA_PADRAO),
            "janela": state.get("learning", {}).get("janela", JANELA_PADRAO),
        },
    )
    return state

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
        
    def _alpha_para_comando(self, cmd: str, alpha_sugerido: float | None = None) -> float:
        """
        α efetivo por comando.

        - /gerar: retorna ALPHA_LOCK_VALUE se LOCK_ALPHA_GERAR=True (lock 0.36).
        - demais: prioriza alpha_sugerido; se None, lê do estado, nesta ordem:
            1) st["alpha"] (raiz)  -> gravado pelo auto_aprender
            2) st["learning"]["alpha"] (legado/compat)
            3) fallback 0.36
        """
        # 1) Lock imediato para /gerar (estabilidade total no front de geração)
        if cmd == "/gerar" and globals().get("LOCK_ALPHA_GERAR", False):
            return float(globals().get("ALPHA_LOCK_VALUE", 0.36))

        # 2) Se o chamador sugeriu um α explícito, respeitamos
        if isinstance(alpha_sugerido, (int, float)):
            try:
                return float(alpha_sugerido)
            except Exception:
                pass  # cai para os próximos métodos de obtenção

        # 3) Tenta obter α “dinâmico” do estado persistente
        alpha_dinamico = 0.36
        try:
            # usa self.st se existir; senão carrega do storage
            st = getattr(self, "st", None) or (_bolao_load_state() or {})

            # (1) raiz primeiro (valor “oficial” aprendido)
            alpha_dinamico = float(st.get("alpha", alpha_dinamico))

            # (2) compatibilidade: learning.alpha (se existir, tem precedência só se definido)
            alpha_dinamico = float((st.get("learning") or {}).get("alpha", alpha_dinamico))
        except Exception:
            alpha_dinamico = 0.36  # fallback hard

        return float(alpha_dinamico)

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
    
    def _media_real_do_lote_persistido(self) -> float:
        """
        Lê o lote definitivo salvo em st["learning"]["last_generation"]["apostas"]
        e calcula a média REAL de acertos contra o último oficial.
        Robusto a falhas: retorna 0.00 se algo der errado.
        """
        try:
            st = _normalize_state_defaults(_bolao_load_state() or {})
            aps = (st.get("learning") or {}).get("last_generation", {}).get("apostas") or []
            oficial = set(self._ultimo_resultado(carregar_historico(HISTORY_PATH)))
            if aps:
                hits = [sum(1 for n in a if n in oficial) for a in aps]
                return round(sum(hits) / float(len(hits)), 2)
            return 0.00
        except Exception:
            return 0.00

    def _latest_snapshot(self) -> _Snapshot:
        historico = carregar_historico(HISTORY_PATH)
        if not historico:
            raise ValueError("Histórico vazio.")
        tamanho = len(historico)
        ultimo = self._ultimo_resultado(historico)
        h8 = _hash_dezenas(ultimo)
        snapshot_id = f"{tamanho}|{h8}"
        return _Snapshot(snapshot_id=snapshot_id, tamanho=tamanho, dezenas=ultimo)
    
    async def _send_long(self, update: Update, text: str, parse_mode: str = "HTML"):
        # quebra por "blocos" separados por linhas em branco
        blocks = text.split("\n\n")
        parts = []
        cur = ""
        for b in blocks:
            if not cur:
                cur = b
            elif len(cur) + 2 + len(b) <= TELEGRAM_SAFE_MAX:
                cur += "\n\n" + b
            else:
                parts.append(cur)
                cur = b
        if cur:
            parts.append(cur)

        # envia cada parte com rodapé (Parte X/Y)
        total = len(parts)
        for i, p in enumerate(parts, 1):
            suffix = f"\n\n<i>Parte {i}/{total}</i>" if total > 1 else ""
            await update.message.reply_text(p + suffix, parse_mode=parse_mode)

    # ------------- Gerador preditivo -------------
    def _gerar_apostas_inteligentes(
        self,
        qtd: int = QTD_BILHETES_PADRAO,
        janela: int = JANELA_PADRAO,
        alpha: float = ALPHA_PADRAO,
    ) -> List[List[int]]:
        """
        Gera bilhetes usando o NOVO preditor unificado (utils.predictor.Predictor).

        - Usa janela e alpha já clampados e alinhados com o restante do bot.
        - Gera um pool maior (4x) e aplica um filtro pós-geração leve:
            • paridade entre 6 e 9 pares (não trava em 7–8 aqui);
            • 1 a 4 dezenas por coluna (matriz 5x5);
            • relaxamento progressivo se necessário.
        - A forma FINAL (paridade 7–8, seq≤3, anti-overlap≤11) continua sendo garantida
          pelos funis de selagem/triplo check depois desse gerador.
        """
        try:
            # garante parâmetros dentro dos limites globais
            qtd, janela, alpha = self._clamp_params(qtd, janela, alpha)

            # carrega histórico e corta para a janela desejada
            historico = carregar_historico(HISTORY_PATH)
            janela_hist = ultimos_n_concursos(historico, janela)

            # filtro leve de composição (não é o TRIPLO CHECK ainda)
            filtro = FilterConfig(
                paridade_min=6,   # deixa respirar: 6–9 pares aqui
                paridade_max=9,
                col_min=1,
                col_max=4,
                relax_steps=2,
            )

            # configura o preditor:
            # - alpha vindo do runtime (/gerar trava em 0.36)
            # - pool maior para ter opções na hora de filtrar
            cfg = GeradorApostasConfig(
                janela=janela,
                alpha=alpha,
                filtro=filtro,
                pool_multiplier=4,   # pool um pouco maior para aumentar diversidade
                # bias_R já vem com default 0.35 no dataclass
            )

            modelo = Predictor(cfg)
            modelo.fit(janela_hist, janela=janela)
            return modelo.gerar_apostas(qtd=qtd)

        except Exception:
            # Fallback duro e bem explícito para não travar o bot
            logger.error(
                "Falha no gerador preditivo; aplicando fallback completamente aleatório.",
                exc_info=True,
            )
            import random
            rng = random.Random()
            return [sorted(rng.sample(range(1, 26), 15)) for _ in range(max(1, qtd))]


    # ========================
    # Aprendizado REAL (revisado c/ política ≥11)
    # ========================
    async def auto_aprender(self, update, context):
        """
        Aprendizado REAL com guard-rails e sem mexer no gerador/estratégia.
        - Gating por política 'official_gate' (ignora se não houver novo oficial, conforme sua política).
        - Usa o último resultado oficial e a última geração salva para medir desempenho REAL.
        - Triplo check-in do lote (paridade 7–8, max_seq<=3, anti-overlap<=11 e sem duplicatas).
        - Corrige cálculo de média; só atualiza α/bias_meta se o lote estiver limpo e consistente.
        - Ajustes estáveis: clamps e decaimento suave (evita “deriva”).
        - Mantém estado se algo falhar (nunca trava o bot).
        - Política do usuário: <11 acertos de média = RUIM (sem reforço). Alvo = 11..15.
        """
        from datetime import datetime
        from zoneinfo import ZoneInfo

        # ---- parâmetros de controle do aprendizado (conservadores) ----
        TARGET_MEDIA = 9.0             # alvo operacional interno da calibração; NÃO é o corte de qualidade
        ALPHA_MIN, ALPHA_MAX = 0.30, 0.42
        ALPHA_STEP = 0.02              # passo de correção de alpha (pequeno)
        MEDIA_GATE = 0.50              # só mexe em alpha se |média - TARGET| >= 0.5

        # bias_meta (R/par/seq) – limites/ganhos
        META_MIN, META_MAX = -0.20, 0.20
        ETA_META = 0.08                # ganho principal para meta-bias
        GAMMA_DECAY = 0.10             # decaimento leve ao zero
        SEQ_PENALTY = 0.05             # penalidade se houver seq>3
        PAR_RANGE = (7, 8)             # paridade alvo por aposta
        OVERLAP_MAX = 11               # anti-overlap entre apostas
        TIMEZONE = globals().get("TIMEZONE", "America/Sao_Paulo")

        # ---- helpers locais (independentes do resto do código) ----
        def _clamp(x, lo, hi): return lo if x < lo else hi if x > hi else x

        def _max_seq_local(sorted_list):
            if not sorted_list:
                return 0
            m = cur = 1
            for a, b in zip(sorted_list, sorted_list[1:]):
                if b == a + 1:
                    cur += 1
                    if cur > m: m = cur
                else:
                    cur = 1
            return m

        def _pares(q):
            return sum(1 for n in q if (n % 2 == 0))

        def _hits(ap, oficial_set):
            return sum(1 for n in ap if n in oficial_set)

        def _overlap(a, b):
            sa, sb = set(a), set(b)
            return len(sa & sb)

        def _triplo_check(apostas, oficial):
            """Retorna (ok, diag) onde ok=True se o lote atende 100% aos 3 critérios."""
            diag = {
                "paridade_falhas": [],
                "seq_falhas": [],
                "overlap_falhas": [],
                "duplicatas": [],
            }
            # Paridade e Sequência por aposta
            for idx, ap in enumerate(apostas, 1):
                p = _pares(ap)
                if not (PAR_RANGE[0] <= p <= PAR_RANGE[1]):
                    diag["paridade_falhas"].append(idx)
                if _max_seq_local(sorted(ap)) > 3:
                    diag["seq_falhas"].append(idx)

            # Anti-overlap + duplicatas
            seen = {}
            for i in range(len(apostas)):
                key = tuple(sorted(apostas[i]))
                seen.setdefault(key, []).append(i + 1)
                for j in range(i + 1, len(apostas)):
                    ov = _overlap(apostas[i], apostas[j])
                    if ov > OVERLAP_MAX:
                        diag["overlap_falhas"].append((i + 1, j + 1, ov))
            # duplicatas
            for k, ids in seen.items():
                if len(ids) > 1:
                    diag["duplicatas"].append(list(ids))

            ok = not (diag["paridade_falhas"] or diag["seq_falhas"] or diag["overlap_falhas"] or diag["duplicatas"])
            return ok, diag

        try:
            # ---------------- Estado e política de gating ----------------
            state = _normalize_state_defaults(_bolao_load_state() or {})
            # Se política exigir novo oficial e não houver sinal de “novo”, apenas retorna
            if state.get("learn_policy", globals().get("LEARN_POLICY", "always")) == "official_gate":
                logger.info("[GATED] auto_aprender ignorado pela política 'official_gate'.")
                return

            # Último oficial (snapshot) e última geração disponível (para medir acertos)
            snap = getattr(self, "_latest_snapshot", lambda: None)()
            ultimo = list(getattr(snap, "dezenas", [])) if snap else []
            if not ultimo:
                logger.info("Sem último resultado oficial disponível; aprendizado não aplicado.")
                return
            oficial_set = set(ultimo)

            last_gen = (state.get("learning", {}) or {}).get("last_generation", {})
            apostas = last_gen.get("apostas") or []
            if not apostas:
                logger.info("Sem apostas em last_generation; nada a aprender neste ciclo.")
                return

            # ---------------- Auditoria mínima (não trava se falhar) ----------------
            try:
                _append_learn_log(getattr(snap, "snapshot_id", "--"), ultimo, apostas)
            except Exception:
                logger.warning("Falha ao registrar learn_log mínimo (seguindo adiante).", exc_info=True)

            # ---------------- Métricas reais do lote ----------------
            acertos_por_aposta = [_hits(ap, oficial_set) for ap in apostas]
            media_real = sum(acertos_por_aposta) / float(len(apostas))

            # Triplo check-in do lote
            ok_lote, diag = _triplo_check(apostas, ultimo)

            # Distribuição de repetição (R = acertos vs oficial)
            from collections import Counter
            cR = Counter(acertos_por_aposta)
            dist_R = {r: cR.get(r, 0) for r in (8, 9, 10, 11)}

            # ---------------- Consistência de média ----------------
            media_consistente = True  # sem outra fonte para comparar aqui

            # ---------------- Política de atualização base ----------------
            alpha_atual = float(state.get("alpha", globals().get("ALPHA_PADRAO", 0.36)))
            janela_atual = int((state.get("learning", {}) or {}).get("janela", globals().get("JANELA_PADRAO", 60)))

            # Não atualiza nada se o lote estiver “sujo”
            if not ok_lote:
                st_learn = state.setdefault("learning", {})
                st_learn["last_learn"] = {
                    "updated_at": datetime.now(ZoneInfo(TIMEZONE)).isoformat(),
                    "media_real": round(media_real, 4),
                    "janela": janela_atual,
                    "diag": diag,
                    "note": "Lote com falhas no triplo check-in; α/bias_meta preservados."
                }
                _bolao_save_state(state)
                logger.info("[Aprendizado] Triplo check-in falhou; parâmetros preservados.")
                return

            if not media_consistente:
                st_learn = state.setdefault("learning", {})
                st_learn["last_learn"] = {
                    "updated_at": datetime.now(ZoneInfo(TIMEZONE)).isoformat(),
                    "media_real": round(media_real, 4),
                    "janela": janela_atual,
                    "diag": diag,
                    "note": "Média inconsistente; α/bias_meta preservados."
                }
                _bolao_save_state(state)
                logger.info("[Aprendizado] Média inconsistente; parâmetros preservados.")
                return

            # ---------------- Ajuste estável de α (proposta) ----------------
            delta_media = media_real - TARGET_MEDIA
            if abs(delta_media) >= MEDIA_GATE:
                alpha_novo = _clamp(alpha_atual + ALPHA_STEP * delta_media, ALPHA_MIN, ALPHA_MAX)
            else:
                alpha_novo = alpha_atual  # sem ajuste se muito perto do alvo interno

            # ---------------- Ajuste estável de bias_meta (R, par, seq) ----------------
            learn = state.setdefault("learning", {})
            bias_meta = learn.get("bias_meta") or {"R": 0.0, "par": 0.0, "seq": 0.0}

            # (a) Repetição (R): alvo Mestre ~ maioria 9R-10R, com 1x8R e 1x11R
            n = float(len(apostas))
            alvo = {8: 1.0 / n, 9: 0.46, 10: 0.46, 11: 1.0 / n}  # ex.: 30 apostas => 1/30, 46%, 46%, 1/30
            obs = {r: dist_R.get(r, 0) / n for r in (8, 9, 10, 11)}
            dR = sum((alvo[r] - obs.get(r, 0.0)) for r in (8, 9, 10, 11))  # erro agregado simples
            bias_meta["R"] = _clamp((1 - GAMMA_DECAY) * bias_meta.get("R", 0.0) + ETA_META * dR, META_MIN, META_MAX)

            # (b) Paridade: fração fora de 7–8 deve tender a 0
            frac_fora = sum(1 for ap in apostas if not (PAR_RANGE[0] <= _pares(ap) <= PAR_RANGE[1])) / n
            bias_meta["par"] = _clamp((1 - GAMMA_DECAY) * bias_meta.get("par", 0.0) + ETA_META * frac_fora,
                                      -0.10, 0.10)

            # (c) Sequência: penaliza se houver QUALQUER seq>3; senão decai ao zero
            any_seq_bad = any(_max_seq_local(sorted(ap)) > 3 for ap in apostas)
            if any_seq_bad:
                bias_meta["seq"] = _clamp(bias_meta.get("seq", 0.0) - SEQ_PENALTY, -0.20, 0.05)
            else:
                bias_meta["seq"] = bias_meta.get("seq", 0.0) * (1 - GAMMA_DECAY)

            # Bias por dezena (decaimento leve)
            bias_num = {int(k): float(v) for (k, v) in (state.get("bias") or {}).items() if str(k).isdigit()}
            for d in range(1, 26):
                bias_num[d] = bias_num.get(d, 0.0) * (1 - 0.05)  # 5% em direção a 0

            # ============================================================
            # >>>>>>> AQUI ENTRA O BLOCO DE POLÍTICA ≥11 (PERSISTÊNCIA + MSG) <<<<<<<
            # (Substitui o trecho antigo que salvava 'state["alpha"] = ...' e montava a mensagem.)
            st = _normalize_state_defaults(_bolao_load_state() or {})
            st = self._coagir_estado_lock_alpha(st)     # garante lock e runtime coerentes
            learn = st.setdefault("learning", {})
            # Classificação pela média real (regra do usuário)
            qualidade = self._classificar_lote_por_media(media_real)

            # Gate por desempenho (RUIM <11 => sem reforço)
            aplicar_reforco = (qualidade != "RUIM")

            # Gates adicionais
            alpha_lock = st["locks"].get("alpha_travado", True)
            official_gate = st["policies"].get("official_gate", True)

            # === Atualização de ALPHA conforme política/lock ===
            if aplicar_reforco:
                if alpha_lock:
                    learn["alpha_proposto"] = float(alpha_novo)
                    msg_alpha = f"α proposto: {alpha_novo:.2f} (pendente; lock ativo)"
                else:
                    if official_gate:
                        learn["alpha"] = float(alpha_novo)
                        learn["alpha_proposto"] = None
                        # Ajusta runtime para refletir o novo valor
                        st["runtime"]["alpha_usado"] = float(alpha_novo)
                        msg_alpha = f"α atualizado (oficial): {alpha_novo:.2f}"
                    else:
                        learn["alpha_proposto"] = float(alpha_novo)
                        msg_alpha = f"α proposto: {alpha_novo:.2f} (aguardando gate oficial)"
            else:
                msg_alpha = f"α mantido: {st['runtime'].get('alpha_usado', ALPHA_LOCK_VALUE):.2f} (lote RUIM; sem reforço)"

            # === Atualização de BIAS conforme política ===
            if aplicar_reforco:
                learn["bias_meta"] = {k: float(v) for k, v in bias_meta.items()}
                st["bias"] = bias_num
                msg_bias = f"bias[R]={bias_meta.get('R', 0.0):+.3f}  bias[par]={bias_meta.get('par', 0.0):+.3f}  bias[seq]={bias_meta.get('seq', 0.0):+.3f}"
            else:
                msg_bias = "bias inalterado (lote RUIM)"

            # === Manter demais campos de aprendizado (históricos/diag) ===
            learn["janela"] = int(janela_atual)
            learn["last_learn"] = {
                "updated_at": datetime.now(ZoneInfo(TIMEZONE)).isoformat(),
                "media_real": round(media_real, 4),
                "delta_media": round((media_real - TARGET_MEDIA), 4),
                "janela": janela_atual,
                "dist_R": {int(k): int(v) for k, v in dist_R.items()},
                "diag": {"ok": aplicar_reforco, "qualidade": qualidade},
            }

            # Persistir estado com segurança
            try:
                _bolao_save_state(st)
            except Exception:
                pass

            # === Mensagem ao usuário — clara quanto à qualidade ===
            alpha_usado_msg = float(st["runtime"].get("alpha_usado", ALPHA_LOCK_VALUE))
            lock_ativo = st["locks"].get("alpha_travado", True)

            if lock_ativo:
                alpha_info = f"α usado: {alpha_usado_msg:.2f} (travado)"
                if learn.get("alpha_proposto") is not None:
                    alpha_info += f" | α proposto: {float(learn['alpha_proposto']):.2f} (pendente)"
            else:
                alpha_info = f"α usado: {alpha_usado_msg:.2f} (livre)"

            msg = (
                "📈 Aprendizado leve atualizado.\n"
                f"• Lote avaliado: {len(apostas)} apostas\n"
                f"• Média de acertos: {media_real:.2f}  →  Qualidade: {qualidade} (alvo ≥ {DESEMPENHO_MINIMO_R:.0f})\n"
                f"• {alpha_info}\n"
                f"• {msg_alpha}\n"
                f"• {msg_bias}"
            )
            await update.message.reply_text(msg)
            # ============================================================

        except Exception:
            logger.warning("Falha no aprendizado real; mantendo parâmetros anteriores.", exc_info=True)

    def _classificar_lote_por_media(self, media_real: float) -> str:
        """
        Classifica a qualidade do lote com base na média real de acertos (R).
        Regra do usuário: <11 = RUIM; alvo = 11..15.
        """
        if media_real < DESEMPENHO_MINIMO_R:
            return "RUIM"
        if media_real < DESEMPENHO_BOM_R:
            return "OK"
        if media_real < 13.0:
            return "BOM"
        return "EXCELENTE"

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
        
        # Comandos “visíveis”
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("gerar", self.gerar_apostas))
        self.app.add_handler(CommandHandler("meuid", self.meuid))
        self.app.add_handler(CommandHandler("autorizar", self.autorizar))
        self.app.add_handler(CommandHandler("remover", self.remover))
        self.app.add_handler(CommandHandler("backtest", self.backtest))
        self.app.add_handler(CommandHandler("mestre", self.mestre))
        self.app.add_handler(CommandHandler("diagbase", self.diagbase))
        self.app.add_handler(CommandHandler("ping", self.ping))
        self.app.add_handler(CommandHandler("versao", self.versao))
        self.app.add_handler(CommandHandler("mestre_bolao", self.mestre_bolao))
        self.app.add_handler(CommandHandler("refinar_bolao", self.refinar_bolao))
        self.app.add_handler(CommandHandler("estado_bolao", self.estado_bolao))
        self.app.add_handler(CommandHandler("bolao20", self.bolao20))
        self.app.add_handler(CommandHandler("conferir_bolao20", self.conferir_bolao20))
        self.app.add_handler(CommandHandler("confirmar", self.confirmar))  # <-- novo comando
        # Handler para comandos desconhecidos (DEVE ficar por último)
        self.app.add_handler(MessageHandler(filters.COMMAND, self._unknown_command))
        logger.info(
            "Handlers ativos: /start /gerar /mestre /mestre_bolao /refinar_bolao "
            "/meuid /autorizar /remover /backtest /diagbase /ping /versao "
            "/estado_bolao /bolao20 /conferir_bolao20 /confirmar + unknown"
        )

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
            "Use /start para ver o menu de comandos disponíveis."
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
        Comando /gerar – Estratégia Mestre, rápido e estável.

        • α TRAVADO = 0.36 (LOCK_ALPHA_GERAR=True), independente do aprendizado.
        • Paridade alvo: 7–8 | Máx. sequência: ≤3 | Anti-overlap: ≤11.
        • Repetição R: foco em 9R–10R, com 1×8R e 1×11R de variação.
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
            pass  # mantém defaults

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
            # Usa 'alpha' já travado via _alpha_para_comando; alpha_usado é o efetivo do runtime
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
    
    
    # =======================[ FINALIZADOR MESTRE ]=======================

    def _dedup_lote(self, apostas: list[list[int]]) -> list[list[int]]:
        seen = set()
        out = []
        for a in apostas:
            t = tuple(sorted(set(a)))
            if t not in seen:
                out.append(list(t))
                seen.add(t)
        return out

    def _freq_dezenas(self, apostas: list[list[int]]) -> dict[int, int]:
        from collections import Counter
        c = Counter()
        for a in apostas:
            c.update(a)
        return dict(c)

    def _aplicar_cap_dezenas(self, apostas: list[list[int]], cap_por_dezena: int, ultimo: list[int]) -> list[list[int]]:
        """
        Se alguma dezena exceder o cap, substitui nas apostas mais 'carregadas'
        por dezenas menos frequentes, mantendo forma (7–8, seq≤3).
        """
        universo = list(range(1, 26))
        freq = self._freq_dezenas(apostas)
        # ordena por excesso (mais usados primeiro)
        excesso = sorted([n for n, f in freq.items() if f > cap_por_dezena], key=lambda n: freq[n], reverse=True)
        if not excesso:
            return apostas

        # candidatos menos usados
        def _menos_usados_atual():
            freq2 = self._freq_dezenas(apostas)
            return sorted(universo, key=lambda n: freq2.get(n, 0))

        # percorre apostas e substitui com cautela
        for idx, a in enumerate(apostas):
            a_set = set(a)
            trocou = False
            for n_ex in excesso:
                if n_ex not in a_set:
                    continue
                # tenta trocar n_ex por um pouco usado que não quebre shape
                for cand in _menos_usados_atual():
                    if cand in a_set:
                        continue
                    # faz a troca temporária
                    b = sorted(set((a_set - {n_ex}) | {cand}))
                    # garante forma
                    try:
                        b = self._hard_lock_fast(b, ultimo=ultimo or [], anchors=frozenset())
                    except Exception:
                        b = self._ajustar_paridade_e_seq(b, alvo_par=(7, 8), max_seq=3)
                    # aceita se continuar com 15 e respeitar pares/seq
                    pares = self._contar_pares(b) if hasattr(self, "_contar_pares") else sum(1 for x in b if x % 2 == 0)
                    if len(b) == 15 and 7 <= pares <= 8 and self._max_seq(b) <= 3:
                        apostas[idx] = b
                        trocou = True
                        break
                if trocou:
                    break
        return apostas

    def _cap_por_par(self, target_qtd: int) -> int:
        # limite leve para coocorrência: metade do número de bilhetes (arredonda p/ cima)
        from math import ceil
        return max(1, ceil(target_qtd / 2))

    def _aplicar_cap_par(self, apostas: list[list[int]], cap_por_par: int, ultimo: list[int]) -> list[list[int]]:
        """
        Se algum par ocorre acima do cap, tenta diluir trocando 1 número do par
        nas apostas mais carregadas por um candidato pouco usado, mantendo a forma.
        """
        from collections import Counter
        from itertools import combinations
        universo = list(range(1, 26))

        # conta pares
        cnt = Counter()
        for a in apostas:
            for i, j in combinations(sorted(a), 2):
                cnt[(i, j)] += 1
        # quais pares estouraram
        viol = [p for p, f in cnt.items() if f > cap_por_par]
        if not viol:
            return apostas

        # tenta diluir
        def _menos_usados_local():
            freq = self._freq_dezenas(apostas)
            return sorted(universo, key=lambda n: freq.get(n, 0))

        for p in sorted(viol, key=lambda x: cnt[x], reverse=True):
            i, j = p
            for idx, a in enumerate(apostas):
                if i in a and j in a:
                    a_set = set(a)
                    # remove um dos dois e insere um candidato menos usado
                    for drop in (i, j):
                        cand_list = _menos_usados_local()
                        for cand in cand_list:
                            if cand in a_set:
                                continue
                            b = sorted(set((a_set - {drop}) | {cand}))
                            try:
                                b = self._hard_lock_fast(b, ultimo=ultimo or [], anchors=frozenset())
                            except Exception:
                                b = self._ajustar_paridade_e_seq(b, alvo_par=(7, 8), max_seq=3)
                            pares = self._contar_pares(b) if hasattr(self, "_contar_pares") else sum(1 for x in b if x % 2 == 0)
                            if len(b) == 15 and 7 <= pares <= 8 and self._max_seq(b) <= 3:
                                apostas[idx] = b
                                break
                        else:
                            continue
                        break
        return apostas

    def _garantir_qtd(self, apostas_ok: list[list[int]], target_qtd: int, ultimo: list[int], call_salt: int, overlap_max: int) -> list[list[int]]:
        # repõe até atingir exatamente target_qtd, respeitando overlap e forma
        from math import inf
        seen = {tuple(sorted(a)) for a in apostas_ok}
        rep_salt = call_salt
        universo = list(range(1, 26))
        while len(apostas_ok) < target_qtd:
            rep_salt += 1
            # usa seu fallback determinístico (já existente em /gerar)
            try:
                cand = self._hard_lock_fast(self._canon(self._fallback(1, rep_salt)[0]), ultimo=ultimo or [], anchors=frozenset())
            except Exception:
                # fallback simples: mistura último + complemento
                L = list(ultimo) or universo[:15]
                C = [n for n in universo if n not in L]
                a = (L[rep_salt % len(L):] + L[:rep_salt % len(L)])[:8] + (C[(rep_salt // 7) % len(C):] + C[:(rep_salt // 7) % len(C)])[:7]
                cand = sorted(set(a))
                try:
                    cand = self._hard_lock_fast(cand, ultimo=ultimo or [], anchors=frozenset())
                except Exception:
                    cand = self._ajustar_paridade_e_seq(cand, alvo_par=(7, 8), max_seq=3)
            t = tuple(sorted(cand))
            if t in seen:
                continue
            # checa overlap com o lote atual
            ok = True
            for b in apostas_ok:
                if len(set(cand) & set(b)) > overlap_max:
                    ok = False
                    break
            if ok:
                apostas_ok.append(sorted(cand))
                seen.add(t)
        return apostas_ok

    # ===================[ FINALIZADOR MESTRE + HELPERS ]===================
    def _finalizar_lote_mestre(
        self,
        apostas: list[list[int]],
        ultimo: list[int],
        target_qtd: int,
        call_salt: int,
        overlap_max: int = 11,
        max_ciclos: int = 6,
        aplicar_cap_par: bool = True,
    ) -> list[list[int]]:
        """
        Funil único de saída:
          1) dedup
          2) selagem de forma (7–8, seq≤3)
          3) anti-overlap global ≤ overlap_max
          4) cap de diversidade (por dezena e, opcionalmente, por par)
          5) re-selagem
          6) triplo check ; se reprovar, repara e repete (até max_ciclos)
          7) garante quantidade exata (repreenche + re-finaliza rápido)
        """
        from math import ceil

        ultimo = ultimo or []
        # caps de diversidade
        # cada dezena não deve aparecer "muito acima" da média teórica (15/25 do total por bilhete)
        cap_dezena = ceil(1.1 * target_qtd * 15 / 25)  # 1.1× média teórica
        cap_pair   = self._cap_por_par(target_qtd)     # limite de coocorrência por par (def abaixo)

        def _shape(a: list[int]) -> list[int]:
            """Trava forma: paridade 7–8 e seq≤3, preservando inteligência quando possível."""
            try:
                b = self._hard_lock_fast(a, ultimo=ultimo or [], anchors=frozenset())
            except Exception:
                b = self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3)
            # normaliza e garante 15
            b = sorted(set(int(x) for x in b if 1 <= int(x) <= 25))
            if len(b) != 15:
                # completa/corta de forma estável
                universo = list(range(1, 26))
                if len(b) < 15:
                    # tenta completar evitando vizinhos pra reduzir corridas
                    comp = [n for n in universo if n not in b]
                    for n in comp:
                        if len(b) == 15:
                            break
                        if (n-1 not in b) and (n+1 not in b):
                            b.append(n)
                    if len(b) < 15:
                        for n in comp:
                            if len(b) == 15:
                                break
                            if n not in b:
                                b.append(n)
                    b = sorted(b)
                else:
                    # corta mantendo ~metade do 'ultimo'
                    u_set = set(ultimo)
                    keep, rest = [], []
                    for n in b:
                        (keep if n in u_set else rest).append(n)
                    b = (keep[:8] + rest)[:15]
                    b = sorted(b)
            # reforço final de forma
            try:
                b = self._hard_lock_fast(b, ultimo=ultimo or [], anchors=frozenset())
            except Exception:
                b = self._ajustar_paridade_e_seq(b, alvo_par=(7, 8), max_seq=3)
            return sorted(set(b))

        # laço de refinamento
        apostas_ok = [sorted(set(x)) for x in (apostas or [])]
        for _ in range(max_ciclos):
            # 1) dedup
            apostas_ok = self._dedup_lote(apostas_ok)

            # 2) shape
            apostas_ok = [_shape(a) for a in apostas_ok]

            # 3) anti-overlap global
            try:
                apostas_ok = self._forcar_anti_overlap(apostas_ok, ultimo=ultimo or [], limite=overlap_max)
            except Exception:
                pass

            # 4) diversidade (caps)
            apostas_ok = self._aplicar_cap_dezenas(apostas_ok, cap_dezena, ultimo)
            if aplicar_cap_par:
                apostas_ok = self._aplicar_cap_par(apostas_ok, cap_pair, ultimo)

            # 5) re-shape
            apostas_ok = [_shape(a) for a in apostas_ok]

            # 6) triplo check
            ok, _diag = self._triplo_check_stricto(apostas_ok)
            if ok:
                break

        # 7) garantir quantidade exata
        if len(apostas_ok) < target_qtd:
            apostas_ok = self._garantir_qtd(apostas_ok, target_qtd, ultimo, call_salt, overlap_max)

        # rodada final idempotente
        apostas_ok = self._dedup_lote(apostas_ok)
        apostas_ok = [_shape(a) for a in apostas_ok]
        try:
            apostas_ok = self._forcar_anti_overlap(apostas_ok, ultimo=ultimo or [], limite=overlap_max)
        except Exception:
            pass
        apostas_ok = [_shape(a) for a in apostas_ok]

        return [sorted(a) for a in apostas_ok]
    # ===================[ FIM FINALIZADOR MESTRE ]===================


    # ===================[ HELPERS DE DIVERSIDADE E SUPORTE ]===================
    def _cap_por_par(self, target_qtd: int) -> int:
        """
        Limite de coocorrência por par (dupla de dezenas) em todo o lote.
        Regra simples e segura: no máximo ~1/3 do total de bilhetes, arredondando pra cima.
        """
        from math import ceil
        return max(1, ceil(target_qtd / 3))


    def _dedup_lote(self, apostas: list[list[int]]) -> list[list[int]]:
        """Remove duplicatas preservando a primeira ocorrência, mantendo ordenação canônica."""
        seen = set()
        res: list[list[int]] = []
        for a in apostas or []:
            t = tuple(sorted(set(a)))
            if t not in seen:
                seen.add(t)
                res.append(list(t))
        return res


    def _aplicar_cap_dezenas(
        self,
        apostas: list[list[int]],
        cap_dezena: int,
        ultimo: list[int] | None = None,
    ) -> list[list[int]]:
        """
        Limita a frequência de cada dezena a <= cap_dezena.
        Tática: identifica dezenas excedentes e tenta trocá-las por dezenas subutilizadas,
        preservando forma via _hard_lock_fast/_ajustar_paridade_e_seq.
        """
        from collections import Counter

        ultimo = ultimo or []
        universo = list(range(1, 26))
        apostas = [sorted(set(a)) for a in (apostas or [])]

        def _shape(a: list[int]) -> list[int]:
            try:
                b = self._hard_lock_fast(a, ultimo=ultimo or [], anchors=frozenset())
            except Exception:
                b = self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3)
            return sorted(set(b))

        for _ in range(3):  # até 3 passes leves
            cnt = Counter(n for a in apostas for n in a)
            excedentes = [n for n, c in cnt.items() if c > cap_dezena]
            if not excedentes:
                break
            # escolhe candidatos subutilizados (menor frequência primeiro)
            faltantes = [n for n in universo if cnt.get(n, 0) < cap_dezena]
            faltantes.sort(key=lambda x: cnt.get(x, 0))
            if not faltantes:
                break

            # para cada dezena excedente, percorre bilhetes contendo-a e tenta trocar
            for n_ex in sorted(excedentes, key=lambda x: -cnt[x]):
                if cnt[n_ex] <= cap_dezena:
                    continue
                for i, a in enumerate(apostas):
                    if n_ex not in a:
                        continue
                    # tenta substituir por uma faltante que não estoure sequência
                    for cand in faltantes:
                        if cand in a:
                            continue
                        # evita vizinho direto para reduzir corridas
                        if (cand - 1 in a) or (cand + 1 in a):
                            continue
                        a2 = sorted(set([x for x in a if x != n_ex] + [cand]))
                        a2 = _shape(a2)
                        if len(a2) == 15 and cand in a2 and n_ex not in a2:
                            apostas[i] = a2
                            cnt[n_ex] -= 1
                            cnt[cand] += 1
                            break
                    if cnt[n_ex] <= cap_dezena:
                        break

        return [sorted(set(a)) for a in apostas]


    def _aplicar_cap_par(
        self,
        apostas: list[list[int]],
        cap_pair: int,
        ultimo: list[int] | None = None,
    ) -> list[list[int]]:
        """
        Limita coocorrência de QUALQUER par {x,y} a <= cap_pair.
        Quando um par estoura o limite, quebramos a coocorrência em alguns bilhetes
        trocando 1 dos 2 números por uma dezena subutilizada e re-selando a forma.
        """
        from collections import Counter
        from itertools import combinations

        ultimo = ultimo or []
        universo = list(range(1, 26))
        apostas = [sorted(set(a)) for a in (apostas or [])]

        def _shape(a: list[int]) -> list[int]:
            try:
                b = self._hard_lock_fast(a, ultimo=ultimo or [], anchors=frozenset())
            except Exception:
                b = self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3)
            return sorted(set(b))

        # conta pares
        def _pair_counts(lote: list[list[int]]) -> Counter:
            c = Counter()
            for a in lote:
                for x, y in combinations(sorted(a), 2):
                    c[(x, y)] += 1
            return c

        for _ in range(2):  # 2 passes leves
            pc = _pair_counts(apostas)
            ruins = [(p, c) for p, c in pc.items() if c > cap_pair]
            if not ruins:
                break
            # ordena piores primeiro
            ruins.sort(key=lambda t: -t[1])

            # mapa de frequência de dezenas para achar substitutos
            from collections import Counter as Cnt
            dz = Cnt(n for a in apostas for n in a)

            for (x, y), c in ruins:
                # para cada bilhete que contém ambas, quebra o par
                for i, a in enumerate(apostas):
                    if x in a and y in a and c > cap_pair:
                        # tenta substituir x OU y por dezena menos usada
                        candidatos = sorted(
                            (n for n in universo if n not in a),
                            key=lambda n: dz.get(n, 0)
                        )
                        trocou = False
                        for sub in candidatos:
                            # escolhe trocar o que mais aparece globalmente para reduzir impacto
                            troca = x if dz.get(x, 0) >= dz.get(y, 0) else y
                            a2 = sorted(set([n for n in a if n != troca] + [sub]))
                            a2 = _shape(a2)
                            if len(a2) == 15 and sub in a2:
                                # atualiza contadores aproximados
                                dz[troca] -= 1
                                dz[sub] += 1
                                apostas[i] = a2
                                c -= 1
                                trocou = True
                                break
                        if not trocou:
                            # fallback: pequena rotação simples
                            fora = [n for n in universo if n not in a]
                            if fora:
                                a2 = sorted(set([n for n in a if n != (x if dz.get(x, 0) >= dz.get(y, 0) else y)] + [fora[0]]))
                                a2 = _shape(a2)
                                if len(a2) == 15:
                                    apostas[i] = a2
                                    c -= 1
                    if c <= cap_pair:
                        break

        return [sorted(set(a)) for a in apostas]


    def _garantir_qtd(
        self,
        apostas: list[list[int]],
        target_qtd: int,
        ultimo: list[int],
        call_salt: int,
        overlap_max: int = 11,
    ) -> list[list[int]]:
        """
        Repõe até atingir target_qtd usando variações determinísticas,
        respeitando anti-overlap e forma Mestre.
        """
        ultimo = ultimo or []
        apostas = [sorted(set(a)) for a in (apostas or [])]
        seen = {tuple(a) for a in apostas}
        rep_salt = int(call_salt)

        def _shape(a: list[int]) -> list[int]:
            try:
                b = self._hard_lock_fast(a, ultimo=ultimo or [], anchors=frozenset())
            except Exception:
                b = self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3)
            return sorted(set(b))

        # gerador determinístico (usa mesma lógica do _fallback do /gerar)
        universo = list(range(1, 26))
        L = list(ultimo) or universo[:15]
        C = [n for n in universo if n not in L]

        def _fallback_one(salt: int) -> list[int]:
            offL = (salt) % len(L)
            offC = ((salt // 7) % len(C)) if C else 0
            a = (L[offL:] + L[:offL])[:8] + (C[offC:] + C[:offC])[:7]
            return _shape(a)

        safety = 0
        while len(apostas) < target_qtd and safety < 2000:
            safety += 1
            rep_salt += 1
            cand = _fallback_one(rep_salt)
            t = tuple(cand)
            if t in seen:
                continue
            # respeita overlap com lote atual
            if all(len(set(cand) & set(b)) <= overlap_max for b in apostas):
                apostas.append(cand)
                seen.add(t)

        # idempotente: dedup + forma + anti-overlap
        apostas = self._dedup_lote(apostas)
        aposta_ok = []
        for a in apostas:
            aposta_ok.append(_shape(a))
        try:
            aposta_ok = self._forcar_anti_overlap(aposta_ok, ultimo=ultimo or [], limite=overlap_max)
        except Exception:
            pass
        aposta_ok = [_shape(a) for a in aposta_ok]
        return [sorted(a) for a in aposta_ok]
    # ===================[ FIM HELPERS ]===================


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
    
    # --- Helper: reparar lote até passar o TRIPLO CHECK ---
    def _reparar_ate_passar_triplo_check(
        self,
        apostas: list[list[int]],
        ultimo: list[int] | None = None,
        limite_overlap: int = 11,
        max_tentativas: int = 5,
    ) -> list[list[int]]:
        """
        Repara o lote garantindo: paridade 7–8, seq≤3, anti-overlap≤limite.
        Estratégia:
          1) Lock de forma aposta a aposta (quebra corridas >3 e ajusta paridade).
          2) Reduz overlap: identifica o par com pior overlap e substitui a aposta mais "conflitante"
             por uma nova variação que:
                - evita a interseção problemática,
                - preserva 7–8 e seq≤3,
                - tenta manter R (repetição) próximo do alvo (9–10).
          3) Repete o ciclo até passar no Triplo Check ou esgotar tentativas.
        """
        from itertools import combinations

        ultimo = ultimo or []
        universo = list(range(1, 26))
        u_set = set(ultimo)

        def _hard_shape(a: list[int]) -> list[int]:
            try:
                b = self._hard_lock_fast(a, ultimo=ultimo, anchors=frozenset())
            except Exception:
                b = self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3)
            b = sorted(set(int(x) for x in b if 1 <= int(x) <= 25))
            # reforço: se por acaso voltou a ter seq longa, aplica ajuste básico
            if self._max_seq(b) > 3 or not (7 <= sum(1 for n in b if n % 2 == 0) <= 8):
                b = self._ajustar_paridade_e_seq(b, alvo_par=(7, 8), max_seq=3)
                b = sorted(set(b))
            return b

        def _overlap(a: list[int], b: list[int]) -> int:
            return len(set(a) & set(b))

        def _worst_pair(l: list[list[int]]) -> tuple[int | None, int | None, int]:
            """retorna (i, j, ovmax) do pior par por overlap; i/j=None se não houver pares."""
            worst = (None, None, -1)
            for i, j in combinations(range(len(l)), 2):
                ov = _overlap(l[i], l[j])
                if ov > worst[2]:
                    worst = (i, j, ov)
            return worst

        def _score_total_overlap(l: list[list[int]], idx: int) -> int:
            return sum(_overlap(l[idx], l[k]) for k in range(len(l)) if k != idx)

        def _nova_variacao(base: list[int], bloquear: set[int], alvo_R=(9, 10)) -> list[int]:
            """
            Gera uma variação determinística evitando 'bloquear' (interseção problemática).
            Mantém ~metade de 'ultimo' (para R≈9–10) e completa com complemento evitando corridas.
            """
            base = [n for n in base if n not in bloquear]
            keep = [n for n in base if n in u_set][:8]  # 7–8 do último
            comp = [n for n in universo if n not in keep]
            # completa evitando vizinhos diretos para reduzir seq
            for n in comp:
                if len(keep) == 15:
                    break
                if (n-1 not in keep) and (n+1 not in keep):
                    keep.append(n)
            if len(keep) < 15:
                for n in comp:
                    if len(keep) == 15:
                        break
                    if n not in keep:
                        keep.append(n)
            return _hard_shape(keep[:15])

        apostas = [sorted(set(a)) for a in apostas]

        for _ in range(max_tentativas):
            # 1) lock de forma em todas
            apostas = [_hard_shape(a) for a in apostas]

            # 2) checa overlap global
            i, j, ov = _worst_pair(apostas)
            if i is None or j is None:  # <<< ajuste de robustez
                break
            if ov <= limite_overlap:
                ok, _ = self._triplo_check_stricto(apostas)
                if ok:
                    return [sorted(a) for a in apostas]
                # mesmo com overlap ok, se ainda reprovar por forma, reforça shape e segue
                apostas = [_hard_shape(a) for a in apostas]
                ok, _ = self._triplo_check_stricto(apostas)
                if ok:
                    return [sorted(a) for a in apostas]
                continue

            # 3) reduzir overlap do pior par
            # escolhe qual substituir: a mais "conflitante" com o lote
            sco_i = _score_total_overlap(apostas, i)
            sco_j = _score_total_overlap(apostas, j)
            trocar = i if sco_i >= sco_j else j

            inter = set(apostas[i]) & set(apostas[j])  # o que está causando o conflito
            base  = apostas[trocar]
            nova  = _nova_variacao(base, bloquear=inter)

            # se, por acaso, nova ficou igual, force um giro simples
            if set(nova) == set(base):
                fora = [n for n in universo if n not in set(base)]
                if fora:
                    nova = sorted(set(list(base)[1:] + fora[:1]))
                    nova = _hard_shape(nova)

            apostas[trocar] = nova
            # volta ao loop (novo ciclo), repetindo lock + checagem

        # se chegou aqui, retorna o melhor possível
        return [sorted(a) for a in apostas]

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
    
    def _overlap_count(self, a: list[int], b: list[int]) -> int:
        return len(set(a) & set(b))

    def _rebuild_candidate(self, base: list[int], ultimo: list[int]) -> list[int]:
        """
        Gera uma variação segura do bilhete 'base' para quebrar overlap alto.
        Mantém shape Mestre (paridade 7–8, seq≤3).
        """
        try:
            universo = list(range(1, 26))
            fora = [n for n in universo if n not in base]

            # Troca até 3 números, priorizando tirar os que estão em clusters (vizinhos)
            troca = []
            for n in base:
                if (n - 1 in base) or (n + 1 in base):
                    troca.append(n)
                if len(troca) == 3:
                    break

            novo = [n for n in base if n not in troca]
            comp = [n for n in fora if n not in novo][:len(troca)]
            novo += comp
            novo = sorted(set(novo))[:15]

            # Reforça shape
            novo = self._hard_lock_fast(novo, ultimo=ultimo or [], anchors=frozenset())
            return sorted(novo)
        except Exception:
            try:
                return self._ajustar_paridade_e_seq(sorted(base), alvo_par=(7, 8), max_seq=3)
            except Exception:
                return sorted(base)
            
    def _candidate_ok(self, cand: list[int], lote: list[list[int]], ultimo: list[int], limite: int) -> bool:
        """Valida paridade 7–8, seq≤3, sem duplicar aposta e overlap ≤ limite com o lote."""
        c = sorted(set(int(x) for x in cand if 1 <= int(x) <= 25))
        if len(c) != 15:
            return False
        # shape
        pares = sum(1 for n in c if n % 2 == 0)
        if not (7 <= pares <= 8):
            return False
        try:
            if self._max_seq(c) > 3:
                return False
        except Exception:
            pass
        # duplicata
        if tuple(c) in {tuple(a) for a in lote}:
            return False
        # overlap
        for a in lote:
            if self._overlap_count(a, c) > limite:
                return False
        return True

    def _refill_to_target(self, apostas: list[list[int]], target_qtd: int, ultimo: list[int], salt_base: int, limite: int) -> list[list[int]]:
        """
        Repõe apostas até atingir target_qtd, respeitando shape Mestre e overlap ≤ limite.
        Usa o fallback determinístico com sal progressivo e selagem final.
        """
        universo = list(range(1, 26))

        def _fallback_unit(salt: int) -> list[int]:
            L = list(ultimo) or universo[:15]
            C = [n for n in universo if n not in L]
            offL = (salt * 3) % len(L)
            offC = (salt * 5) % len(C) if C else 0
            a = (L[offL:] + L[:offL])[:8] + (C[offC:] + C[:offC])[:7]
            try:
                a = self._hard_lock_fast(a, ultimo=ultimo or [], anchors=frozenset())
            except Exception:
                a = self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3)
            return sorted(set(a))[:15]

        salt = max(1, int(salt_base))
        guard = 0
        while len(apostas) < target_qtd and guard < 400:
            guard += 1
            salt += 1
            cand = _fallback_unit(salt)
            # pequena diversificação: tente uma segunda reconstrução se bater em overlap
            if not self._candidate_ok(cand, apostas, ultimo, limite):
                cand = self._rebuild_candidate(cand, ultimo=ultimo or [])
            if self._candidate_ok(cand, apostas, ultimo, limite):
                # selagem teimosa final
                try:
                    cand = self._hard_lock_fast(cand, ultimo=ultimo or [], anchors=frozenset())
                except Exception:
                    cand = self._ajustar_paridade_e_seq(cand, alvo_par=(7, 8), max_seq=3)
                apostas.append(sorted(cand))
        return apostas

    def _forcar_anti_overlap(
        self,
        apostas: list[list[int]],
        ultimo: list[int],
        limite: int = 11,
    ) -> list[list[int]]:
        """
        Garante: sem duplicatas e overlap(a_i, a_j) ≤ limite,
        priorizando DESCOLAR pares hiper-coocorrentes sem quebrar a forma Mestre
        (15 únicos, paridade 7–8, seq≤3).

        Estratégia:
          - Calcula coocorrência de pares no lote.
          - Identifica pares "hiper" (aparecem muitas vezes).
          - Sempre que um par de apostas ultrapassa o overlap 'limite':
              * escolhe para ajuste a aposta com MAIOR custo interno
                (mais pares hiper + mais dezenas marcadas como ruído, se houver).
              * troca 1–2 dezenas que colidem por complementares de baixo custo
                (pouca coocorrência com a outra aposta e pouco adjacentes).
          - Repite em poucos ciclos até todos overlaps ficarem ≤ limite ou não haver mais melhoria.
        """
        if len(apostas) < 2:
            return [sorted(set(x)) for x in apostas]

        from collections import Counter

        universo = list(range(1, 26))

        # -------- 1) mapa de coocorrência atual --------
        pair_cnt: Counter[tuple[int, int]] = Counter()
        for a in apostas:
            s = sorted(set(a))
            for i in range(len(s)):
                for j in range(i + 1, len(s)):
                    p = (s[i], s[j])
                    pair_cnt[p] += 1

        # Pares "hiper" = aparecem em pelo menos ~metade do lote
        limite_par = max(2, int(0.5 * len(apostas)))
        hiper = {p for p, c in pair_cnt.items() if c >= limite_par}

        # Ruídos opcionais globais (se você tiver definido RUIDOS em algum lugar)
        try:
            ruidos = set(int(x) for x in globals().get("RUIDOS", set()))
        except Exception:
            ruidos = set()

        def _custo_lista(a: list[int]) -> int:
            """
            Custo interno da aposta:
              - +1 para cada par hiper presente
              - +1 para cada dezena marcada como ruído
            """
            c = 0
            s = set(a)
            arr = sorted(a)
            for i in range(len(arr)):
                for j in range(i + 1, len(arr)):
                    if (arr[i], arr[j]) in hiper:
                        c += 1
            c += sum(1 for n in s if n in ruidos)
            return c

        def _overlap(a: list[int], b: list[int]) -> int:
            return len(set(a) & set(b))

        # Normaliza apostas de entrada
        aps = [sorted(set(x)) for x in apostas]

        # -------- 2) laço de descolamento global --------
        for _ in range(8):  # limite de ciclos para não rodar demais
            # encontra pior par (i, j) pelo maior overlap acima do limite
            worst = None
            worst_ov = limite
            for i in range(len(aps)):
                for j in range(i + 1, len(aps)):
                    ov = _overlap(aps[i], aps[j])
                    if ov > worst_ov:
                        worst_ov = ov
                        worst = (i, j)

            if not worst:
                # não há mais pares violando o limite
                break

            i, j = worst

            # decide qual aposta “descolar”: a de maior custo interno
            pick = i if _custo_lista(aps[i]) >= _custo_lista(aps[j]) else j
            keep = j if pick == i else i

            s_pick = set(aps[pick])
            s_keep = set(aps[keep])

            # dezenas que colidem entre as duas
            colisores = sorted(list(s_pick & s_keep))  # determinístico
            if not colisores:
                continue

            # candidatos complementares para substituir
            comps = [n for n in universo if n not in s_pick]

            # ranqueia complementares: queremos minimizar coocorrência com o "keep"
            comps_scored: list[tuple[int, int]] = []
            for c in comps:
                score = 0
                # penaliza formar pares hiper com elementos do keep
                for x in s_keep:
                    p = (min(c, x), max(c, x))
                    if p in hiper:
                        score += 2
                # leve penalização se adjacente a muitos da própria aposta (evita seq longas)
                for x in s_pick:
                    if abs(c - x) == 1:
                        score += 1
                comps_scored.append((score, c))
            comps_scored.sort()  # menor score = melhor

            changed_local = False
            # troca no máximo 2 dezenas por ciclo nessa aposta
            for rem in colisores[:2]:
                for _, add in comps_scored:
                    if add in s_pick:
                        continue
                    b = sorted((s_pick - {rem}) | {add})

                    # re-sela forma (paridade 7–8, seq≤3, 15 únicas)
                    try:
                        b = self._hard_lock_fast(b, ultimo=ultimo or [], anchors=frozenset())
                    except Exception:
                        b = self._ajustar_paridade_e_seq(b, alvo_par=(7, 8), max_seq=3)

                    if not self._shape_ok_basico(b):
                        continue

                    # checa se essa nova aposta respeita overlap com todo mundo
                    if all(
                        _overlap(b, x) <= limite
                        for x in aps
                        if x is not aps[pick]
                    ):
                        aps[pick] = sorted(set(b))
                        changed_local = True
                        break
                if changed_local:
                    break

            # se não conseguiu mudar nada nesse par, passa para o próximo ciclo
            # (outros pares podem ser descolados na próxima iteração)

        # saída normalizada
        return [sorted(set(a)) for a in aps]

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
        st = _normalize_state_defaults(_bolao_load_state() or {})
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
        st = _normalize_state_defaults(_bolao_load_state() or {})
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
    

    # =========================
    # Pós-filtro determinístico (dentro da classe)
    # =========================
    # Corrige apostas que excedem:
    #   - Anti-overlap > 11 (com o resultado oficial anterior)
    #   - Sequência consecutiva > 3
    # Totalmente determinístico (zero aleatoriedade) e idempotente.

    def _pfu_anti_overlap_count(self, aposta: List[int], anterior: List[int]) -> int:
        s = set(aposta)
        return sum(1 for n in anterior if n in s)

    def _pfu_seq_max(self, ap: List[int]) -> int:
        if not ap:
            return 0
        ap_sorted = sorted(ap)
        mx = cur = 1
        for i in range(1, len(ap_sorted)):
            if ap_sorted[i] == ap_sorted[i - 1] + 1:
                cur += 1
                if cur > mx:
                    mx = cur
            else:
                cur = 1
        return mx

    def _pfu_primeiro_cluster_maior_que(self, ap: List[int], limite: int) -> Optional[tuple[int, int]]:
        """
        Retorna (inicio, tamanho) do primeiro cluster consecutivo com tamanho > limite.
        """
        ap_sorted = sorted(ap)
        if not ap_sorted:
            return None
        start = ap_sorted[0]
        cur = 1
        for i in range(1, len(ap_sorted)):
            if ap_sorted[i] == ap_sorted[i - 1] + 1:
                cur += 1
            else:
                if cur > limite:
                    return (start, cur)
                start = ap_sorted[i]
                cur = 1
        if cur > limite:
            return (start, cur)
        return None

    def _pfu_melhor_substituto_deterministico(
        self,
        candidatos: List[int],
        aposta: List[int],
        anterior: List[int],
        preferir_quebrar_seq: bool = True
    ) -> Optional[int]:
        """
        Escolhe o menor número 'n' que:
          1) não está em 'aposta';
          2) NÃO pertence ao 'anterior' (para reduzir overlap), quando possível;
          3) (opcional) evita criar sequência com dois lados.
        """
        ap_set = set(aposta)
        ant_set = set(anterior)

        # Preferência: fora do anterior e sem criar seq “de três lados”
        for n in sorted(candidatos):
            if n in ap_set or n in ant_set:
                continue
            if preferir_quebrar_seq and (n - 1 in ap_set) and (n + 1 in ap_set):
                continue
            return n

        # Fallback: se só restaram candidatos no anterior, pegue o melhor possível
        for n in sorted(candidatos):
            if n in ap_set:
                continue
            if preferir_quebrar_seq and (n - 1 in ap_set) and (n + 1 in ap_set):
                continue
            return n

        return None

    def _pfu_reparar_aposta_overlap_seq(
        self,
        aposta: List[int],
        resultado_anterior: Optional[List[int]],
        overlap_max: int = 11,
        seq_max: int = 3
    ) -> List[int]:
        """
        Ajustes mínimos e determinísticos:
          - Se não houver 'resultado_anterior', NO-OP.
          - Se overlap > overlap_max: substitui números presentes no anterior.
          - Se seq_max estourar: quebra o primeiro cluster > seq_max.
        Mantém tamanho=15 e ordenação crescente.
        """
        if not resultado_anterior:
            return sorted(aposta)

        ap = sorted(aposta)
        universo = list(range(1, 26))
        anterior = sorted(resultado_anterior)
        ant_set = set(anterior)

        # pool de números livres (não presentes na aposta)
        pool_livres = [n for n in universo if n not in ap]

        # 1) Corrigir overlap excessivo
        while self._pfu_anti_overlap_count(ap, anterior) > overlap_max:
            # Remove um número que esteja no anterior — maior primeiro (determinístico)
            removivel = None
            for n in sorted([x for x in ap if x in ant_set], reverse=True):
                removivel = n
                break
            if removivel is None:
                break  # não há como reduzir

            sub = self._pfu_melhor_substituto_deterministico(pool_livres, ap, anterior, preferir_quebrar_seq=True)
            if sub is None:
                break

            ap.remove(removivel)
            ap.append(sub)
            ap.sort()

            pool_livres.remove(sub)
            pool_livres.append(removivel)
            pool_livres.sort()

        # 2) Corrigir sequência excessiva
        while self._pfu_seq_max(ap) > seq_max:
            cluster = self._pfu_primeiro_cluster_maior_que(ap, seq_max)
            if not cluster:
                break
            inicio, tam = cluster
            remover = inicio + tam - 1  # remove o MAIOR do cluster (determinístico)

            sub = self._pfu_melhor_substituto_deterministico(pool_livres, ap, anterior, preferir_quebrar_seq=True)
            if sub is None:
                break

            ap.remove(remover)
            ap.append(sub)
            ap.sort()

            pool_livres.remove(sub)
            pool_livres.append(remover)
            pool_livres.sort()

        return ap

    def _pfu_obter_resultado_anterior_seguro(self) -> Optional[List[int]]:
        """
        Retorna o resultado oficial imediatamente anterior ao atual, se disponível.
        Nunca lança exceções.
        """
        try:
            if hasattr(self, "_resultado_anterior"):
                r = self._resultado_anterior()
                if r and isinstance(r, (list, tuple)) and len(r) == 15:
                    return list(map(int, r))
            if hasattr(self, "_penultimo_resultado"):
                r = self._penultimo_resultado()
                if r and isinstance(r, (list, tuple)) and len(r) == 15:
                    return list(map(int, r))
            # Fallback via state
            st_try = _bolao_load_state() or {}
            off = (st_try.get("official") or {})
            penultimo = off.get("penultimo") or off.get("anterior") or []
            if penultimo and isinstance(penultimo, (list, tuple)) and len(penultimo) == 15:
                return list(map(int, penultimo))
        except Exception:
            pass
        return None

    def _pos_filtro_unificado_deterministico(
        self,
        apostas: List[List[int]],
        overlap_limite: Optional[int] = None,
        seq_limite: Optional[int] = None
    ) -> List[List[int]]:
        """
        Filtro final determinístico do pipeline de pós-processamento.
        Busca limites no state quando não informados:
          - overlap_limite: default 11
          - seq_limite: default 3
        Se não encontrar o 'resultado anterior', retorna as apostas como estão (NO-OP).
        """
        st = _normalize_state_defaults(_bolao_load_state() or {})
        learn = (st.get("learning") or {})
        overlap_max = int(overlap_limite if overlap_limite is not None else learn.get("last_overlap_limit", 11))
        seq_max = int(seq_limite if seq_limite is not None else learn.get("max_seq", 3))

        anterior = self._pfu_obter_resultado_anterior_seguro()
        if not anterior:
            return [sorted(ap) for ap in apostas]

        ajustadas = []
        for ap in apostas:
            ap_fix = self._pfu_reparar_aposta_overlap_seq(
                ap,
                resultado_anterior=anterior,
                overlap_max=overlap_max,
                seq_max=seq_max
            )
            ajustadas.append(ap_fix)

        return ajustadas
    
    # ==== Auditoria de Lote (contagem, coocorrência, triplo check, reforço) ====

    PAR_RANGE = (7, 8)
    OVERLAP_MAX = 11
    ANCHOR_SET = frozenset({9, 11})      # exemplo de âncoras leves já usadas no projeto
    ANCHOR_SCALE = 0.5                   # mesma escala reduzida aplicada às âncoras

    def _pares(ap):
        return sum(1 for n in ap if n % 2 == 0)

    def _max_seq_local(ap_ordenada):
        # máximo de sequência contígua (ex.: [5,6,7] => 3)
        if not ap_ordenada: return 0
        m = cur = 1
        for i in range(1, len(ap_ordenada)):
            if ap_ordenada[i] == ap_ordenada[i-1] + 1:
                cur += 1
                m = max(m, cur)
            else:
                cur = 1
        return m

    def _overlap(a, b):
        sa, sb = set(a), set(b)
        return len(sa & sb)

    # --- TRIPLO CHECK (stricto) — retorna DICT para consumo interno ---
    def _triplo_check_stricto(self,
                              apostas: list[list[int]],
                              alvo_par=(7, 8),
                              max_seq: int = 3,
                              max_overlap: int | None = None) -> tuple[bool, dict]:
        """
        Valida o LOTE inteiro:
          • Cada aposta: 15 dezenas únicas entre 1..25, paridade 7–8, sequência máxima ≤3
          • Lote: sem duplicatas exatas e overlap(a_i, a_j) ≤ max_overlap
        Retorna: (ok_lote: bool, diag_dict: {paridade_falhas, seq_falhas, overlap_falhas, duplicatas})
        """
        if max_overlap is None:
            max_overlap = int(globals().get("BOLAO_MAX_OVERLAP", 11))

        def _max_seq_local(nums: list[int]) -> int:
            s = sorted(nums)
            run = best = 1
            for i in range(1, len(s)):
                if s[i] == s[i-1] + 1:
                    run += 1
                    if run > best:
                        best = run
                else:
                    run = 1
            return best

        def _overlap(a: list[int], b: list[int]) -> int:
            return len(set(a) & set(b))

        diag = {
            "paridade_falhas": [],   # [idx_aposta]
            "seq_falhas": [],        # [idx_aposta]
            "overlap_falhas": [],    # [(i,j,ov)]
            "duplicatas": []         # [[ids...], ...]
        }

        # valida por aposta
        seen = {}
        for idx, a in enumerate(apostas, 1):
            a = sorted(set(int(n) for n in a if 1 <= int(n) <= 25))
            if len(a) != 15:
                # se quiser sinalizar tamanho errado, use seq_falhas para “travar”
                diag["seq_falhas"].append(idx)
            pares = sum(1 for n in a if n % 2 == 0)
            if not (alvo_par[0] <= pares <= alvo_par[1]):
                diag["paridade_falhas"].append(idx)
            if _max_seq_local(a) > max_seq:
                diag["seq_falhas"].append(idx)
            t = tuple(a)
            seen.setdefault(t, []).append(idx)

        # duplicatas
        for _, ids in seen.items():
            if len(ids) > 1:
                diag["duplicatas"].append(ids)

        # overlaps
        norm = [sorted(set(a)) for a in apostas]
        for i in range(len(norm)):
            for j in range(i + 1, len(norm)):
                ov = _overlap(norm[i], norm[j])
                if ov > max_overlap:
                    diag["overlap_falhas"].append((i + 1, j + 1, ov))

        ok_lote = not (diag["paridade_falhas"] or diag["seq_falhas"] or diag["overlap_falhas"] or diag["duplicatas"])
        return ok_lote, diag


    def _formatar_triplo_check_diag(self, diag: dict, max_overlap: int | None = None) -> str:
        """Gera um HTML compacto a partir do diag-dict do _triplo_check_stricto."""
        if max_overlap is None:
            max_overlap = int(globals().get("BOLAO_MAX_OVERLAP", 11))

        linhas = ["<b>🔎 TRIPLO CHECK (stricto)</b>"]
        if diag.get("paridade_falhas"):
            linhas.append(f"• Paridade fora de 7–8 nas apostas: {diag['paridade_falhas']}")
        else:
            linhas.append("• Paridade: ✅ todas em 7–8")
        if diag.get("seq_falhas"):
            linhas.append(f"• Sequência >3 nas apostas: {diag['seq_falhas']}")
        else:
            linhas.append("• Sequência: ✅ todas com seq≤3")
        if diag.get("overlap_falhas"):
            worst = max(diag["overlap_falhas"], key=lambda t: t[2]) if diag["overlap_falhas"] else None
            if worst:
                i, j, ov = worst
                linhas.append(f"• Overlap máximo: ❌ {ov} (> {max_overlap}) entre Aposta {i:02d} e {j:02d}")
        else:
            linhas.append(f"• Overlap máximo: ✅ ≤ {max_overlap}")
        if diag.get("duplicatas"):
            linhas.append(f"• Duplicatas detectadas: ❌ {diag['duplicatas']}")
        else:
            linhas.append("• Duplicatas: ✅ nenhuma")
        return "\n".join(linhas)

    def _coocorrencias(apostas):
        """Retorna contagem de pares (i,j) que apareceram juntos no lote."""
        c = Counter()
        for ap in apostas:
            for a, b in combinations(sorted(set(ap)), 2):
                c[(a, b)] += 1
        return c

    def _hits_por_aposta(apostas, oficial_set):
        return [sum(1 for n in ap if n in oficial_set) for ap in apostas]

    def _reward_penalty(apostas, oficial_set):
        """Mapa (dezena -> delta) com recompensa por HIT e penalização por MISS; âncoras com escala reduzida."""
        freq = Counter()
        for ap in apostas:
            for n in ap:
                freq[n] += 1

        deltas = defaultdict(float)
        for n, k in freq.items():
            if n in oficial_set:
                base = 1.0
            else:
                base = -1.0
            if n in ANCHOR_SET:
                base *= ANCHOR_SCALE
            deltas[n] += base * k
        return dict(deltas)

    def _format_dez(l):
        return " ".join(f"{n:02d}" for n in l)

    # ==== Helpers de coerência de estado (alpha/lock) ====

    def _ensure_keys_safe(self, st: dict) -> dict:
        """Garante chaves mínimas sem sobrescrever valores existentes."""
        st = st or {}
        st.setdefault("runtime", {})
        st.setdefault("learning", {})
        st.setdefault("locks", {})
        st.setdefault("policies", {})

        # locks
        st["locks"].setdefault("alpha_travado", bool(globals().get("LOCK_ALPHA_GERAR", True)))

        # runtime
        st["runtime"].setdefault("alpha_usado", float(ALPHA_LOCK_VALUE))

        # learning
        st["learning"].setdefault("alpha", float(ALPHA_LOCK_VALUE))         # legado
        st["learning"].setdefault("alpha_proposto", None)                   # proposta pendente
        st["learning"].setdefault("last_overlap_limit", 11)
        st["learning"].setdefault("max_seq", 3)

        # policies
        st["policies"].setdefault("official_gate", True)

        return st

    def _coagir_estado_lock_alpha(self, st: dict) -> dict:
        """
        Se lock ativo: força o alpha_usado em runtime = ALPHA_LOCK_VALUE e
        NÃO deixa 'learning.alpha' substituir o usado. Se detectar divergência
        entre learning.alpha e o lock, move para 'alpha_proposto'.
        """
        st = self._ensure_keys_safe(st)
        lock_ativo = st["locks"].get("alpha_travado", True)

        if lock_ativo:
            # alpha efetivo de geração
            st["runtime"]["alpha_usado"] = float(ALPHA_LOCK_VALUE)

            # Se o valor em learning.alpha divergir, não aplicamos — vira proposta
            try:
                alpha_legado = float(st["learning"].get("alpha", ALPHA_LOCK_VALUE))
            except Exception:
                alpha_legado = float(ALPHA_LOCK_VALUE)

            if abs(alpha_legado - float(ALPHA_LOCK_VALUE)) > 1e-9:
                st["learning"]["alpha_proposto"] = alpha_legado
                # Mantemos learning.alpha igual ao lock para não confundir ler em outros pontos
                st["learning"]["alpha"] = float(ALPHA_LOCK_VALUE)
        else:
            # Sem lock, o alpha usado segue o 'learning.alpha'
            try:
                st["runtime"]["alpha_usado"] = float(st["learning"].get("alpha", ALPHA_LOCK_VALUE))
            except Exception:
                st["runtime"]["alpha_usado"] = float(ALPHA_LOCK_VALUE)

        return st

    def _alpha_para_execucao(self, st: dict) -> float:
        """
        Retorna o alpha que deve ser usado na geração AGORA, respeitando o lock.
        """
        st = self._coagir_estado_lock_alpha(st)
        return float(st["runtime"].get("alpha_usado", ALPHA_LOCK_VALUE))

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
        """
        Seleciona a matriz-19 base do Modo Bolão v5 a partir do histórico recente,
        considerando:
          - repetidores fortes do último resultado,
          - ausentes quentes (baixo atraso),
          - neutros balanceados na faixa BOLAO_NEUTRA_RANGE,
          - âncoras obrigatórias (BOLAO_ANCHORS),
        com paridade/seq controladas depois nos estágios seguintes.

        Também sanitiza 'bias' do estado para ignorar chaves não numéricas
        (por exemplo 'R', 'paridade', etc.) para não quebrar.
        """
        if not historico:
            raise ValueError("Histórico vazio.")

        # Último resultado conhecido
        ultimo = self._ultimo_resultado(historico)
        u_set = set(ultimo)

        # --- SANITIZA bias: aceita só dezenas 1..25 como chave ---
        st = _normalize_state_defaults(_bolao_load_state() or {})
        bias_raw = st.get("bias", {})
        bias: dict[int, float] = {}
        if isinstance(bias_raw, dict):
            for k, v in bias_raw.items():
                try:
                    ki = int(k)
                    if 1 <= ki <= 25:
                        bias[ki] = float(v)
                except Exception:
                    # ignora chave suja como "R", "seq", etc.
                    continue

        # --- Janela recente e métricas de apoio ---
        jan_rf = self._janela_recent_first(historico, BOLAO_JANELA)
        freq_eff = _freq_window(jan_rf, bias=bias)
        atrasos = _atrasos_recent_first(jan_rf)

        # 1) Top repetidores do último resultado ("r10")
        #    Ordena pelo freq_eff decrescente e depois pelo número.
        r10 = sorted(ultimo, key=lambda n: (-freq_eff[n], n))[:10]

        # 2) Ausentes "quentes": não saíram no último, baixo atraso
        ausentes = [n for n in range(1, 26) if n not in u_set]
        hot_abs = [n for n in ausentes if atrasos[n] <= 8]
        hot_abs.sort(key=lambda n: (atrasos[n], -freq_eff[n], n))
        # pega até 6 quentes; se não tiver tanto quente, pega ~5
        hot_take = hot_abs[:6] if len(hot_abs) >= 6 else hot_abs[:max(0, 5)]

        # 3) Completa até 19 números com neutros balanceados
        usados = set(r10) | set(hot_take)
        faltam = 19 - len(usados)

        neutrals_pool = [n for n in ausentes if n not in usados]

        def score(n: int):
            """
            Critério para escolher neutros:
            - distância da faixa BOLAO_NEUTRA_RANGE (quanto mais central, melhor)
            - frequência efetiva (mais freq primeiro)
            - atraso (menos atraso primeiro)
            - valor numérico estável
            """
            lo, hi = BOLAO_NEUTRA_RANGE
            if lo <= n <= hi:
                dist = 0
            else:
                dist = min(abs(n - lo), abs(n - hi))
            return (dist, -freq_eff[n], atrasos[n], n)

        neutrals_pool.sort(key=score)
        neutros = neutrals_pool[:max(0, faltam)]

        # 4) Monta matriz inicial
        matriz = sorted(set(r10) | set(hot_take) | set(neutros))

        # 5) Força âncoras (BOLAO_ANCHORS) a entrarem
        for anc in BOLAO_ANCHORS:
            if anc not in matriz:
                # tenta remover um número que:
                # - não é âncora
                # - não está no último resultado (para não "quebrar" R)
                candidatos = [n for n in matriz if n not in BOLAO_ANCHORS and n not in u_set]
                if not candidatos:
                    # se não tiver, remove qualquer não-âncora
                    candidatos = [n for n in matriz if n not in BOLAO_ANCHORS]
                rem = max(
                    candidatos,
                    key=lambda n: (atrasos[n], -freq_eff[n], n),
                    default=None,
                )
                if rem is not None and rem != anc:
                    try:
                        matriz.remove(rem)
                    except ValueError:
                        pass
                    matriz.append(anc)

        # 6) Ajuste final de tamanho = 19
        matriz = sorted(set(matriz))
        if len(matriz) != 19:
            # completa ou corta
            pool = [n for n in range(1, 26) if n not in matriz]
            for n in pool:
                matriz.append(n)
                if len(matriz) == 19:
                    break
            matriz = sorted(matriz)[:19]

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
    
    # ---------- Matriz 20 a partir do Mestre (último + quentes + ausentes fortes) ----------
    def _matriz20_mestre(self, historico: list[list[int]]) -> list[int]:
        """
        Constrói 20 dezenas determinísticas para o bolão:
        - parte do último oficial (15)
        - escolhe 5 ausentes por score: (freq_janela - atraso_norm + bias)
        - sela para manter diversidade (sem viciar)
        """
        if not historico:
            # fallback seguro (nunca quebra)
            return list(range(1, 21))

        ultimo = self._ultimo_resultado(historico)           # 15 dezenas
        hist_win = ultimos_n_concursos(historico, BOLAO20_JANELA)
        if not HISTORY_ORDER_DESC:
            hist_win = list(reversed(hist_win))

        # frequência e atraso
        freq = _freq_window(hist_win, bias=_normalize_state_defaults(_bolao_load_state()).get("bias"))
        atraso = _atrasos_recent_first(hist_win)

        # universo e ausentes do último
        U = set(range(1, 26))
        last_set = set(ultimo)
        ausentes = sorted(list(U - last_set))

        # score determinístico
        # normaliza atraso para [0,1] (quanto menor atraso, maior score)
        max_at = max(atraso.values()) if atraso else 1
        def score(n: int) -> float:
            at = atraso.get(n, max_at)
            at_norm = 1.0 - (at / float(max_at or 1))
            return float(freq.get(n, 0.0)) + 0.75 * at_norm

        # top-5 ausentes por score (sem encadear sequências grandes)
        cand = sorted(ausentes, key=score, reverse=True)
        pick = []
        for n in cand:
            if len(pick) == 5:
                break
            # evita criar correntes longas ao juntar ao último
            if ((n-1) in last_set and (n+1) in last_set):
                continue
            pick.append(n)

        # se faltar, completa pelos próximos
        i = 0
        while len(pick) < 5 and i < len(cand):
            if cand[i] not in pick:
                pick.append(cand[i])
            i += 1

        matriz20 = sorted(list(last_set | set(pick)))[:20]
        return matriz20

    def _fechamento20_reduzido(self, matriz20: list[int], qtd: int, ultimo: list[int]) -> list[list[int]]:
        """
        Gera 'qtd' apostas de 15 dezenas a partir de 'matriz20' (20 dezenas),
        garantindo SEMPRE:
          - cada jogo é SUBCONJUNTO de matriz20 (nunca injeta fora)
          - paridade 7–8
          - SeqMax ≤ 3
        Estratégia:
          - particiona m em 5 grupos de ~4 (ordem determinística)
          - cada jogo exclui 5 números (4 de um grupo + 1 rotativo do próximo)
          - corrige paridade/seq por trocas apenas com as reservas (subset-safe)
          - reintenta variações determinísticas do 5º excluído até fechar requisitos
          - repõe jogos perdidos no dedup para sempre retornar 'qtd'
        """
        # --- saneamento da matriz ---
        m = sorted({int(x) for x in matriz20 if 1 <= int(x) <= 25})
        if len(m) != 20:
            U = [n for n in range(1, 26)]
            for x in U:
                if x not in m:
                    m.append(x)
                if len(m) == 20:
                    break
            m = sorted(m[:20])

        # --- grupos determinísticos por snapshot ---
        snap = self._latest_snapshot()
        seed = (self._stable_hash_int(snap.snapshot_id) % (10**9)) or 1
        order = sorted(m, key=lambda x: (seed ^ (x * 1315423911)) & 0xFFFFFFFF)
        grupos = [order[i::5] for i in range(5)]  # 5 grupos

        def max_seq(arr: list[int]) -> int:
            s = 1
            best = 1
            arrs = sorted(arr)
            for i in range(1, len(arrs)):
                if arrs[i] == arrs[i-1] + 1:
                    s += 1
                    if s > best:
                        best = s
                else:
                    s = 1
            return best

        def pares(arr: list[int]) -> int:
            return sum(1 for n in arr if n % 2 == 0)

        def seal_subset(base: list[int], reserva: list[int]) -> list[int]:
            """
            Ajusta base (subset de m) para paridade 7–8 e SeqMax≤3,
            trocando APENAS com números da 'reserva' (também subset de m).
            Nunca injeta número fora de m.
            """
            base_set = set(base)
            resv = [n for n in reserva if n not in base_set]
            base = sorted(base)

            # 0) blindagem (teórica) contra fora de m
            for x in list(base):
                if x not in m and resv:
                    y = resv.pop(0)
                    base_set.remove(x)
                    base_set.add(y)
                    base = sorted(base_set)

            # 1) Paridade alvo 7–8
            target_low, target_high = 7, 8
            p = pares(base)

            def swap_for_parity(want_even: bool) -> bool:
                nonlocal base, base_set, resv, p
                if want_even:
                    base_out = next((x for x in base if x % 2 == 1), None)
                    res_in  = next((y for y in resv if y % 2 == 0), None)
                    if base_out is None or res_in is None:
                        return False
                else:
                    base_out = next((x for x in base if x % 2 == 0), None)
                    res_in  = next((y for y in resv if y % 2 == 1), None)
                    if base_out is None or res_in is None:
                        return False
                resv.remove(res_in)
                base_set.remove(base_out)
                base_set.add(res_in)
                base = sorted(base_set)
                p = pares(base)
                return True

        # ---- continue seal_subset ----
            guard = 0
            while (p < target_low or p > target_high) and guard < 12:
                guard += 1
                if p < target_low:
                    if not swap_for_parity(want_even=True):
                        break
                elif p > target_high:
                    if not swap_for_parity(want_even=False):
                        break

            # 2) Sequência máxima ≤ 3
            def break_runs() -> bool:
                nonlocal base, base_set, resv
                arr = sorted(base)
                runs = []
                start = 0
                for i in range(1, len(arr)+1):
                    if i == len(arr) or arr[i] != arr[i-1] + 1:
                        runs.append(arr[start:i])
                        start = i
                worst = max(runs, key=len)
                if len(worst) <= 3:
                    return False
                out_idx = len(worst) // 2
                to_remove = worst[out_idx]
                # tenta reserva que não cole em vizinhos
                for y in list(resv):
                    if (y-1 in base_set) or (y+1 in base_set):
                        continue
                    resv.remove(y)
                    base_set.remove(to_remove)
                    base_set.add(y)
                    base = sorted(base_set)
                    return True
                if resv:
                    y = resv.pop(0)
                    base_set.remove(to_remove)
                    base_set.add(y)
                    base = sorted(base_set)
                    return True
                return False

            guard = 0
            while max_seq(base) > 3 and guard < 12:
                guard += 1
                if not break_runs():
                    break

            # 3) Repassa paridade após quebrar sequências (quebra pode alterar p)
            p = pares(base)
            guard = 0
            while (p < target_low or p > target_high) and guard < 6:
                guard += 1
                if p < target_low:
                    if not swap_for_parity(want_even=True):
                        break
                elif p > target_high:
                    if not swap_for_parity(want_even=False):
                        break
                p = pares(base)

            return sorted(base)

        def build_one(k: int, extra_offset: int) -> list[int] | None:
            """
            Constrói o jogo k variando deterministicamente o 5º excluído
            via 'extra_offset' para dar flexibilidade de paridade e seq.
            Retorna None se não conseguiu fechar requisitos.
            """
            g_idx = k % 5
            excl = set(grupos[g_idx])  # 4 números do grupo g_idx
            prox = (g_idx + 1) % 5
            alt = grupos[prox]
            extra = alt[(k + extra_offset) % len(alt)]
            excl.add(extra)

            base = [n for n in m if n not in excl]   # 20-5=15
            reserva = sorted(list(excl))
            base = seal_subset(base, reserva)

            # requisitos
            if max_seq(base) > 3:
                return None
            p = pares(base)
            if not (7 <= p <= 8):
                return None
            return sorted(base)

        # --- geração com reintentos determinísticos por jogo ---
        jogos = []
        seen = set()
        max_try_per_game = 12
        for k in range(max(1, int(qtd))):
            chosen = None
            for off in range(max_try_per_game):
                cand = build_one(k, off)
                if cand is None:
                    continue
                t = tuple(cand)
                if t in seen:
                    continue
                chosen = cand
                break
            # fallback: se ainda não escolheu, aceita 1 duplicado e ajusta depois
            if chosen is None:
                for off in range(max_try_per_game, max_try_per_game + 8):
                    cand = build_one(k, off)
                    if cand is not None:
                        chosen = cand
                        break
            if chosen is None:
                # >>> CORREÇÃO: fallback garante 15 dezenas e subset de m <<<
                base_all = sorted(m)
                shift = (k % 5)
                excl = set(base_all[shift::5][:5])     # exclui 5 espaçados
                base = [n for n in base_all if n not in excl][:15]
                chosen = seal_subset(base, sorted(list(excl)))

            t = tuple(chosen)
            if t not in seen:
                seen.add(t)
                jogos.append(chosen)

        # --- se o dedup natural deixou com menos que 'qtd', gera variações até completar ---
        def _alt_build(index_seed: int) -> list[int] | None:
            """
            Estratégia alternativa: varia o 5º excluído em DOIS grupos vizinhos
            e também permuta 1 número leve na base, tudo determinístico.
            """
            for ghop in (1, 2):  # saltos de grupo para variar reserva
                for off in range(24):
                    g_idx = index_seed % 5
                    excl = set(grupos[g_idx])
                    alt = grupos[(g_idx + ghop) % 5]
                    extra = alt[(index_seed + off) % len(alt)]
                    excl.add(extra)

                    base = [n for n in m if n not in excl]
                    reserva = sorted(list(excl))
                    cand = seal_subset(base, reserva)
                    if max_seq(cand) > 3:
                        continue
                    p = sum(1 for x in cand if x % 2 == 0)
                    if not (7 <= p <= 8):
                        continue

                    t = tuple(sorted(cand))
                    if t in seen:
                        # micro-variação: tenta 1 swap base<->reserva mantendo regras
                        for r in reserva:
                            if r in cand:
                                continue
                            for b in cand:
                                if b in reserva:
                                    continue
                                cand2 = sorted((y if y != b else r) for y in cand)
                                if tuple(cand2) in seen:
                                    continue
                                if max_seq(cand2) <= 3 and 7 <= sum(1 for xx in cand2 if xx % 2 == 0) <= 8:
                                    return cand2
                        continue
                    return sorted(cand)
            return None

        guard_global = 0
        while len(jogos) < qtd and guard_global < 200:
            guard_global += 1
            idx = len(jogos) + guard_global  # muda semente determinística
            cand = _alt_build(idx)
            if cand:
                t = tuple(cand)
                if t not in seen:
                    seen.add(t)
                    jogos.append(cand)

        # garantia final: se ainda faltar (caso extremo), gera micro-variações dentro da matriz até completar
        idx = 0
        while len(jogos) < qtd and idx < 300:
            idx += 1
            base_all = sorted(m)
            shift = (idx % 5)
            excl = set(base_all[shift::5][:5])           # 5 excluídos espaçados
            cand = [n for n in base_all if n not in excl][:15]
            cand = seal_subset(cand, sorted(list(excl)))
            if max_seq(cand) <= 3 and 7 <= sum(1 for x in cand if x % 2 == 0) <= 8:
                t = tuple(sorted(cand))
                if t not in seen:
                    seen.add(t)
                    jogos.append(sorted(cand))

        return [sorted(a) for a in jogos[:qtd]]

    # ---------- Comando público /bolao20 ----------
    async def bolao20(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Uso: /bolao20 [A|B|C]
          A = 10 jogos (14 frequente, não garantido)
          B = 20 jogos (14 muito frequente)  <-- recomendado / estilo lotérica
          C = 36–48 jogos (garantia técnica de 14 se errar ≤1)
        """
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id

        # --- autorização + anti-abuso ---
        if not self._usuario_autorizado(user_id):
            return await update.message.reply_text("⛔ Você não está autorizado a gerar apostas.")
        if not self._is_admin(user_id):
            if _is_temporarily_blocked(user_id):
                return await update.message.reply_text("🚫 Você está temporariamente bloqueado por excesso de tentativas.")
            allowed, warn = _register_command_event(user_id, is_unknown=False)
            if not allowed:
                return await update.message.reply_text(warn)
            if warn:
                await update.message.reply_text(warn)
        if self._hit_cooldown(chat_id, "bolao20"):
            return await update.message.reply_text("⏳ Aguardando cooldown… tente novamente em alguns segundos.")

        # --- plano escolhido ---
        plano = "B"
        if context.args:
            p = context.args[0].strip().upper()
            if p in BOLAO20_PLANOS:
                plano = p

        qtd = BOLAO20_PLANOS[plano]

        # --- histórico e último oficial ---
        historico = carregar_historico(HISTORY_PATH)
        ultimo = self._ultimo_resultado(historico) if historico else []

        # --- autoconferência automática de sessões pendentes ---
        conferidas = self._bolao20_try_autoconferir()
        if conferidas:
            await update.message.reply_text(
                f"ℹ️ {len(conferidas)} sessão(ões) do Bolão 20 conferida(s) automaticamente com o último resultado."
            )

        # --- Matriz 20 determinística + fechamento reduzido ---
        matriz20 = self._matriz20_mestre(historico)
        jogos = self._fechamento20_reduzido(matriz20, qtd=qtd, ultimo=ultimo)

        # --- montagem da mensagem ---
        linhas = [f"🎰 <b>Bolão Matriz 20</b> — Plano <b>{plano}</b> ({BOLAO20_DESCR[plano]})\n"]
        linhas.append(f"<b>Matriz 20:</b> {' '.join(f'{n:02d}' for n in matriz20)}\n")

        for i, a in enumerate(jogos, 1):
            pares = self._contar_pares(a)
            seq   = self._max_seq(a)
            linhas.append(
                f"<b>Jogo {i:02d}:</b> {' '.join(f'{n:02d}' for n in a)}\n"
                f"   🔢 Pares: {pares} | Ímpares: {15 - pares} | SeqMax: {seq}\n"
            )

        # --- rodapé telemétrico ---
        from datetime import datetime
        from zoneinfo import ZoneInfo
        now_sp = datetime.now(ZoneInfo(TIMEZONE)).strftime("%Y-%m-%d %H:%M:%S %Z")
        linhas.append(
            f"<i>janela={BOLAO20_JANELA} | α={BOLAO20_ALPHA:.2f} | jogos={qtd} | {now_sp}</i>"
        )

        # --- registra sessão para conferência posterior ---
        sid = self._bolao20_register_session(
            plano=plano,
            matriz20=matriz20,
            jogos=jogos,
            snapshot_id=self._latest_snapshot().snapshot_id,
            concurso=tentar_descobrir_concurso(carregar_historico(HISTORY_PATH)) if "tentar_descobrir_concurso" in globals() else None
        )
        linhas.append(f"<i>SID:</i> {sid}")

        # --- envia ao usuário ---
        await self._send_long(update, "\n".join(linhas), parse_mode="HTML")

        # ---------- Persistência leve do bolão ----------
    def _bolao20_state_load(self) -> dict:
        try:
            with open(BOLAO20_STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"sessions": []}

    def _bolao20_state_save(self, state: dict) -> None:
        try:
            os.makedirs(os.path.dirname(BOLAO20_STATE_PATH), exist_ok=True)
            with open(BOLAO20_STATE_PATH, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception:
            pass  # falha silenciosa não deve quebrar o fluxo

    def _bolao20_register_session(self, plano: str, matriz20: list[int], jogos: list[list[int]], snapshot_id: str, concurso: int | None) -> str:
        state = self._bolao20_state_load()
        sid = f"{snapshot_id}|{plano}|{int(time.time())}"
        state["sessions"].append({
            "sid": sid,
            "plano": plano,
            "matriz20": jogos and sorted(set(matriz20)) or [],
            "jogos": jogos,
            "snapshot_id": snapshot_id,
            "concurso_base": concurso,     # concurso do snapshot (se souber)
            "checked": False,
            "result_concurso": None,
            "result_dezenas": None,
            "hist": None,                  # preenchido na conferência
        })
        self._bolao20_state_save(state)
        return sid

    # ---------- Contagem de acertos ----------
    @staticmethod
    def _count_hits(aposta: list[int], resultado: list[int]) -> int:
        s = set(aposta)
        return sum(1 for n in resultado if n in s)

    def _conferir_jogos(self, jogos: list[list[int]], resultado: list[int]) -> dict:
        """
        Retorna histograma de acertos e lista de índices por faixa.
        Ex.: {"15": [idxs...], "14": [...], "13": [...], ...}
        """
        hist = {str(k): [] for k in range(6, 16)}  # 6..15 p/ análise
        for i, a in enumerate(jogos, 1):
            h = self._count_hits(a, resultado)
            if str(h) in hist:
                hist[str(h)].append(i)
        return hist

    # ---------- Conferência automática (opcional) ----------
    def _bolao20_try_autoconferir(self) -> list[dict]:
        """
        Se BOLAO20_AUTO_MATCH_LAST=True:
          - pega 'sessions' pendentes (checked=False)
          - cruza com o último resultado oficial
          - grava o histograma de acertos
        Retorna lista de sessões conferidas no ciclo.
        """
        if not BOLAO20_AUTO_MATCH_LAST:
            return []
        state = self._bolao20_state_load()
        if not state.get("sessions"):
            return []

        historico = carregar_historico(HISTORY_PATH)
        if not historico:
            return []

        ultimo = self._ultimo_resultado(historico)
        checked = []

        for s in state["sessions"]:
            if s.get("checked"):
                continue
            jogos = s.get("jogos") or []
            if not jogos:
                continue
            hist = self._conferir_jogos(jogos, ultimo)
            s["checked"] = True
            s["result_concurso"] = tentar_descobrir_concurso(historico) if "tentar_descobrir_concurso" in globals() else None
            s["result_dezenas"] = ultimo
            s["hist"] = hist
            checked.append(s)

        if checked:
            self._bolao20_state_save(state)
        return checked
    
    # ---------- Comando público /conferir_bolao20 ----------
    async def conferir_bolao20(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Uso:
          /conferir_bolao20               -> confere sessões pendentes com o último oficial
          /conferir_bolao20 SID           -> confere/mostra sessão específica
          /conferir_bolao20 concurso=3531 -> confere com resultado de um concurso específico (se disponível)
        """
        user_id = update.effective_user.id
        if not self._usuario_autorizado(user_id):
            return await update.message.reply_text("⛔ Você não está autorizado.")

        args = " ".join(context.args) if context.args else ""
        target_sid = None
        target_concurso = None

        # parse simples dos argumentos
        if "concurso=" in args:
            try:
                target_concurso = int(args.split("concurso=")[1].strip().split()[0])
            except Exception:
                target_concurso = None
        elif args:
            target_sid = args.strip()

        # carrega estado
        state = self._bolao20_state_load()
        sessions = state.get("sessions", [])

        # define resultado alvo (último ou de concurso específico, se você tiver essa função/histórico)
        historico = carregar_historico(HISTORY_PATH)
        if not historico:
            return await update.message.reply_text("Sem histórico para conferir.")

        if target_concurso and "resultado_por_concurso" in globals():
            resultado = resultado_por_concurso(historico, target_concurso)  # opcional no seu projeto
            if not resultado:
                return await update.message.reply_text(f"Não achei resultado do concurso {target_concurso}.")
        else:
            resultado = self._ultimo_resultado(historico)

        # filtra sessões
        if target_sid:
            sessions = [s for s in sessions if s.get("sid") == target_sid]
            if not sessions:
                return await update.message.reply_text("SID não encontrado.")
        else:
            # pega as pendentes para conferência automática
            pend = [s for s in sessions if not s.get("checked")]
            sessions = pend or sessions[-3:]  # mostra últimas 3 se não houver pendentes

        # confere e monta resposta
        linhas = []
        for s in sessions:
            jogos = s.get("jogos") or []
            hist = self._conferir_jogos(jogos, resultado)

            s["checked"] = True
            s["result_concurso"] = target_concurso or s.get("result_concurso")
            s["result_dezenas"] = resultado
            s["hist"] = hist

            linhas.append(f"🧾 <b>SID:</b> {s['sid']}")
            linhas.append(f"Plano: <b>{s.get('plano')}</b> | Jogos: {len(jogos)}")
            linhas.append(f"Resultado: {' '.join(f'{n:02d}' for n in resultado)}")
            # resumo 15/14 (e 13 para fins de telemetria)
            q15 = len(hist.get("15", []))
            q14 = len(hist.get("14", []))
            q13 = len(hist.get("13", []))
            linhas.append(f"🏆 <b>15 acertos:</b> {q15}  |  ⭐ <b>14 acertos:</b> {q14}  |  13 acertos: {q13}")
            if q14 or q15:
                top = []
                if q15:
                    top.append("15→ " + ", ".join(f"#{i:02d}" for i in hist["15"]))
                if q14:
                    top.append("14→ " + ", ".join(f"#{i:02d}" for i in hist["14"]))
                linhas.append("• " + " | ".join(top))
            linhas.append("")  # linha em branco

        self._bolao20_state_save(state)
        if not linhas:
            linhas = ["Não há sessões para conferir."]
        await self._send_long(update, "\n".join(linhas), parse_mode="HTML")

    # --- /mestre: pacote Mestre determinístico selado e sem duplicata ---
    async def mestre(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Gera 10 apostas Mestre (foco repetição alta 9R–10R, +1 variação 8R e +1 variação 11R)
        sempre obedecendo:
          - Paridade 7–8
          - SeqMax ≤ 3
          - anti-overlap ≤ BOLAO_MAX_OVERLAP
          - Dedup (não repetir aposta igual no lote)
          - Cobertura distribuída dos ausentes

        Esse comando é fixo/determinístico para o mesmo último resultado + mesmo chat/user.
        Ele usa o mesmo pipeline de selagem que o Bolão (hard_lock + dedup + anti-overlap + bias),
        para impedir exatamente os problemas que vimos (paridade 4/11, SeqMax 5+, clones).
        """

        user_id = update.effective_user.id
        chat_id = update.effective_chat.id

        # --- autorização ---
        if not self._usuario_autorizado(user_id):
            return await update.message.reply_text("⛔ Você não está autorizado a gerar apostas.")

        # --- anti-abuso / rate limit ---
        if not self._is_admin(user_id):
            if _is_temporarily_blocked(user_id):
                return await update.message.reply_text("🚫 Você está temporariamente bloqueado por excesso de tentativas.")
            allowed, warn = _register_command_event(user_id, is_unknown=False)
            if not allowed:
                return await update.message.reply_text(warn)
            if warn:
                await update.message.reply_text(warn)

        # --- cooldown por chat pra evitar spam humano apertando várias vezes seguidas ---
        if self._hit_cooldown(chat_id, "/mestre"):
            return await update.message.reply_text("⏳ Aguarde alguns segundos antes de pedir novamente.")

        # --- coleta histórico e último resultado ---
        try:
            historico = carregar_historico(HISTORY_PATH)
        except Exception:
            historico = []
        if not historico:
            return await update.message.reply_text("Histórico indisponível.")

        ultimo = self._ultimo_resultado(historico)  # lista ordenada 15 dezenas
        u_set = set(ultimo)
        comp = [n for n in range(1, 26) if n not in u_set]

        # seed estável (user+chat+último resultado)
        mestre_seed = self._calc_mestre_seed(user_id, chat_id, ultimo)

        # ------------------------------------------------------------------
        # 1. Construção bruta das 10 apostas Mestre com metas de repetição R
        #    Alvo do Mestre segundo nossa regra salva em memória:
        #    - Prioriza 9R–10R
        #    - Inclui 1 jogo ~8R e 1 jogo ~11R
        # ------------------------------------------------------------------
        planos_R_base = [10, 9, 9, 10, 10, 9, 10, 8, 11, 10]
        # se mudar essa lista no futuro, manter tamanho 10

        brutas = []
        L = list(ultimo)
        C = comp[:]

        # Geração determinística controlada por offsets fixos no seed
        for idx, r_alvo in enumerate(planos_R_base):
            off_last = (mestre_seed + idx * 3) % max(1, len(L))
            off_comp = (mestre_seed // 7 + idx * 5) % max(1, len(C) if C else 1)

            base_aposta = self._construir_aposta_por_repeticao(
                last_sorted=L,
                comp_sorted=C,
                repeticoes=r_alvo,
                offset_last=off_last,
                offset_comp=off_comp,
            )

            # força paridade 7–8 e seq≤3 logo cedo
            base_aposta = self._hard_lock_fast(
                base_aposta,
                ultimo,
                anchors=frozenset(),           # Mestre não tem âncora fixa obrigatória
                alvo_par=(7, 8),
                max_seq=3,
            )
            brutas.append(sorted(base_aposta))

        # ------------------------------------------------------------------
        # 2. Pós-filtro unificado EXACTO do bolão:
        #    - quebra sequência grande
        #    - ajusta paridade
        #    - dedup forte
        #    - anti-overlap ≤ BOLAO_MAX_OVERLAP
        #    - aplica bias leve
        # ------------------------------------------------------------------
        try:
            refinadas = self._pos_filtro_unificado(brutas, ultimo)
        except Exception:
            logger.warning("Falha em _pos_filtro_unificado dentro /mestre; aplicando fallback mínimo.", exc_info=True)
            # fallback mínimo: aplica hard_lock individual + dedup básico
            refinadas = [self._hard_lock_fast(a, ultimo, anchors=frozenset(), alvo_par=(7, 8), max_seq=3) for a in brutas]
            # dedup básico (remove clones idênticos mantendo primeiros)
            seen_local = set()
            uniq_tmp = []
            for a in refinadas:
                t = tuple(sorted(a))
                if t not in seen_local:
                    seen_local.add(t)
                    uniq_tmp.append(sorted(a))
            refinadas = uniq_tmp

        # ------------------------------------------------------------------
        # 3. Garantia final:
        #    - força novamente forma (7–8 pares, SeqMax ≤3)
        #    - corta/ordena e elimina duplicadas finais
        # ------------------------------------------------------------------
        apostas_finais = []
        seen_final = set()
        for a in refinadas:
            a2 = self._hard_lock_fast(a, ultimo, anchors=frozenset(), alvo_par=(7, 8), max_seq=3)
            t = tuple(sorted(a2))
            if t not in seen_final:
                seen_final.add(t)
                apostas_finais.append(sorted(a2))

        # se por algum motivo ficamos com menos que 10 depois da dedup agressiva,
        # não inventamos aposta aleatória nova (mantemos determinístico).
        # Apenas seguimos com o que sobrou.

        # ------------------------------------------------------------------
        # 4. Telemetria pra cada aposta (pares, ímpares, SeqMax, R)
        # ------------------------------------------------------------------
        telems = [self._telemetria(a, ultimo, alvo_par=(7, 8), max_seq=3) for a in apostas_finais]

        # ------------------------------------------------------------------
        # 5. Registrar geração para aprendizado leve (estado interno)
        try:
            self._registrar_geracao(apostas_finais, base_resultado=ultimo)
        except Exception:
            logger.warning("Falha ao registrar geração (mestre).", exc_info=True)

        # 5.1) >>> NOVO: auditoria CSV (aprendizado REAL)
        try:
            snap = self._latest_snapshot()
            _append_learn_log(snap.snapshot_id, ultimo or [], apostas_finais)
        except Exception:
            logger.warning("Falha para append no learn_log após /mestre.", exc_info=True)

        # ------------------------------------------------------------------
        # 6. Formatar resposta pro usuário
        # ------------------------------------------------------------------
        linhas = ["🎰 <b>SUAS APOSTAS INTELIGENTES — Preset Mestre</b> 🎰\n"]
        ok_count = 0
        for i, (a, t) in enumerate(zip(apostas_finais, telems), start=1):
            status = "✅ OK" if t.ok_total else "🛠️ REPARAR"
            linhas.append(
                f"<b>Aposta {i}:</b> {' '.join(f'{n:02d}' for n in a)}\n"
                f"🔢 Pares: {t.pares} | Ímpares: {t.impares} | SeqMax: {t.max_seq} | {t.repeticoes}R | {status}\n"
            )
            if t.ok_total:
                ok_count += 1

        linhas.append(
            f"\nConformidade: <b>{ok_count}/{len(apostas_finais)}</b> dentro de (paridade 7–8, seq≤3)"
        )
        linhas.append(
            f"<i>Regras: paridade 7–8, seq≤3, anti-overlap≤{BOLAO_MAX_OVERLAP}</i>"
        )

        if SHOW_TIMESTAMP:
            now_sp = datetime.now(ZoneInfo(TIMEZONE))
            carimbo = now_sp.strftime("%Y-%m-%d %H:%M:%S %Z")
            try:
                snap = self._latest_snapshot()
                snap_id = snap.snapshot_id
            except Exception:
                snap_id = "--"
            linhas.append(
                f"<i>base=último resultado | snapshot={snap_id} | tz={TIMEZONE} | {carimbo}</i>"
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
          - Aprendizado automático leve preservando forma

        Aceita 15 dezenas no comando:
            /refinar_bolao 01 02 ... 25
        """
        from datetime import datetime
        from zoneinfo import ZoneInfo
        import traceback

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
            st = _normalize_state_defaults(_bolao_load_state() or {})
            st = dict(st) if isinstance(st, dict) else {}
            raw_bias = st.get("bias", {}) or {}

            bias = {}
            for k, v in raw_bias.items():
                try:
                    ki = int(k)
                    bias[ki] = float(v)
                except Exception:
                    continue

            hits_map = {}
            for k, v in (st.get("hits", {}) or {}).items():
                try:
                    hits_map[int(k)] = int(v)
                except Exception:
                    continue

            seen_map = {}
            for k, v in (st.get("seen", {}) or {}).items():
                try:
                    seen_map[int(k)] = int(v)
                except Exception:
                    continue

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

            # Atualiza bias dezena-a-dezena respeitando âncoras
            for n in mset:
                delta = float(BOLAO_BIAS_HIT) if (n in of_set) else float(BOLAO_BIAS_MISS)
                if n in anch:
                    delta *= float(BOLAO_BIAS_ANCHOR_SCALE)
                bias[n] = _clamp(
                    float(bias.get(n, 0.0)) + delta,
                    float(BOLAO_BIAS_MIN),
                    float(BOLAO_BIAS_MAX),
                )

            # persiste bias/hits/seen/snapshot
            st["bias"] = {int(k): float(v) for k, v in bias.items()}
            st["hits"] = hits_map
            st["seen"] = seen_map
            try:
                st["last_snapshot"] = snap.snapshot_id
            except Exception:
                st["last_snapshot"] = "--"
            _bolao_save_state(st)

            # 4) Matriz19 DEPOIS do refino (já refletindo bias recém-atualizado)
            matriz19_depois = self._selecionar_matriz19(historico)

            # 5) Regenera 19→15 com seed incremental por snapshot (variabilidade determinística)
            try:
                seed_nova = self._next_draw_seed(snap.snapshot_id)
            except Exception:
                seed_nova = self._next_draw_seed("fallback")

            apostas_brutas = self._subsets_19_para_15(matriz19_depois, seed=seed_nova)

            # 6) Selagem rápida por aposta (hard_lock_fast aplica paridade 7–8, seq<=3, âncoras)
            apostas_seladas = [
                self._hard_lock_fast(a, oficial, anchors=frozenset(anchors_tuple))
                for a in apostas_brutas
            ]

            # 7) Pós-filtro unificado (forma + dedup/overlap + bias)
            try:
                apostas_filtradas = self._pos_filtro_unificado(apostas_seladas, ultimo=oficial)
            except Exception:
                logger.warning("pos_filtro_unificado falhou no /refinar_bolao; aplicando hard_lock por aposta.", exc_info=True)
                apostas_filtradas = [
                    self._hard_lock_fast(a, oficial, anchors=frozenset(anchors_tuple))
                    for a in apostas_seladas
                ]

            # 7.1) Validação final teimosa ("belt and suspenders")
            def _canon_local(a: list[int]) -> list[int]:
                a = [int(x) for x in a if 1 <= int(x) <= 25]
                a = sorted(set(a))
                if len(a) < 15:
                    comp = [n for n in range(1, 26) if n not in a]
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
                elif len(a) > 15:
                    a = sorted(a)[:15]
                return a

            apostas_ok = []
            for a in apostas_filtradas[:5]:
                a = _canon_local(a)
                a = self._hard_lock_fast(a, oficial, anchors=frozenset(anchors_tuple))
                apostas_ok.append(a)
            apostas = apostas_ok

            # 7.2) REGISTRO (núcleo + CSV) + NOVO: pending_batches
            try:
                # 1) Estado persistente – núcleo
                self._registrar_geracao(apostas, base_resultado=oficial or [])
            except Exception:
                logger.warning("Falha ao registrar geração para aprendizado leve (/refinar_bolao).", exc_info=True)

            # 2) CSV de auditoria externa (aprendizado REAL)
            try:
                _append_learn_log(snap.snapshot_id, oficial or [], apostas)
            except Exception:
                logger.warning("Falha para append no learn_log após /refinar_bolao.", exc_info=True)

            # >>> NOVO: registrar o lote no estado (pending_batches)
            try:
                st2 = _normalize_state_defaults(_bolao_load_state() or {})
                batches = st2.get("pending_batches", [])
                batches.append({
                    "ts": datetime.now(ZoneInfo(TIMEZONE)).isoformat(),
                    "snapshot": getattr(self._latest_snapshot(), "snapshot_id", "--"),
                    "alpha": float(st2.get("alpha", ALPHA_PADRAO)),
                    "janela": int((st2.get("learning") or {}).get("janela", JANELA_PADRAO)),
                    "oficial_base": " ".join(f"{n:02d}" for n in (oficial or [])),
                    "qtd": len(apostas),
                    "apostas": [" ".join(f"{x:02d}" for x in a) for a in apostas],
                })
                st2["pending_batches"] = batches[-100:]  # mantém histórico curto
                _bolao_save_state(st2)
            except Exception:
                logger.warning("Falha ao registrar pending_batch no /refinar_bolao.", exc_info=True)
            # <<< NOVO

            # 8) Telemetria, placar e resposta
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
                try:
                    snap_id = snap.snapshot_id
                except Exception:
                    snap_id = "--"
                carimbo = datetime.now(ZoneInfo(TIMEZONE)).strftime("%Y-%m-%d %H:%M:%S %Z")
                linhas.append(
                    f"\n<i>snapshot={snap_id} | seed={seed_nova} | tz={TIMEZONE} | /refinar_bolao | {carimbo}</i>"
                )

            linhas.append(f"<i>Regras: paridade 7–8, seq≤3, anti-overlap≤{BOLAO_MAX_OVERLAP}</i>")

            # envia resposta final ao usuário
            texto_final = "\n".join(linhas)
            await update.message.reply_text(texto_final, parse_mode="HTML")

            # 9) Rodar auto_aprender pós-refino (não travar se falhar)
            try:
                await self.auto_aprender(update, context)
            except Exception:
                logger.warning("auto_aprender falhou pós-/refinar_bolao; prosseguindo normalmente.", exc_info=True)

            return

        except Exception as e:
            logger.error("Erro no /refinar_bolao:\n" + traceback.format_exc())
            return await update.message.reply_text(f"Erro no /refinar_bolao: {e}")
        
   # ===================[ UTILITÁRIOS STRICTO ]===================

    def _shape_ok_basico(self, a: list[int]) -> bool:
        """15 únicos, paridade 7–8, seq<=3."""
        if len(a) != 15 or len(set(a)) != 15:
            return False
        pares = sum(1 for n in a if n % 2 == 0)
        if not (7 <= pares <= 8):
            return False
        return self._max_seq(a) <= 3


    def _quebrar_seq_maior_3(self, a: list[int]) -> list[int]:
        """
        Quebra runs >3 trocando o elemento 'central' do run por um candidato seguro,
        tentando preservar paridade 7–8. Determinístico pelo ordenamento.
        """
        a = sorted(set(int(x) for x in a if 1 <= int(x) <= 25))
        if len(a) != 15:
            # normaliza tamanho se necessário (mantém determinismo simples)
            universo = list(range(1, 26))
            for n in universo:
                if n not in a:
                    a.append(n)
                    if len(a) == 15:
                        break
            a = sorted(a)

        # Detecta runs contínuos
        runs = []
        start = 0
        for i in range(1, len(a) + 1):
            if i == len(a) or a[i] != a[i - 1] + 1:
                runs.append((start, i - 1))
                start = i

        # Para cada run > 3, substitui o elemento central por um “safe”
        universo = set(range(1, 26))
        usados = set(a)
        candidatos_base = [n for n in sorted(universo - usados)]
        pares_alvo_min, pares_alvo_max = 7, 8

        for (i0, i1) in runs:
            tamanho = i1 - i0 + 1
            if tamanho > 3:
                idx = i0 + tamanho // 2  # posição central
                # tenta inserir candidato não adjacente aos vizinhos
                for add in candidatos_base:
                    if (add - 1 not in a) and (add + 1 not in a):
                        tmp = a[:]
                        tmp[idx] = add
                        tmp = sorted(set(tmp))
                        pares = sum(1 for n in tmp if n % 2 == 0)
                        if (7 <= pares <= 8) and (self._max_seq(tmp) <= 3):
                            a = tmp
                            usados = set(a)
                            candidatos_base = [n for n in sorted(universo - usados)]
                            break
                else:
                    # fallback: aceita o primeiro candidato que respeite forma
                    for add in candidatos_base:
                        tmp = a[:]
                        tmp[idx] = add
                        tmp = sorted(set(tmp))
                        pares = sum(1 for n in tmp if n % 2 == 0)
                        if (7 <= pares <= 8) and (self._max_seq(tmp) <= 3):
                            a = tmp
                            usados = set(a)
                            candidatos_base = [n for n in sorted(universo - usados)]
                            break

        # selagem final curta
        try:
            a = self._hard_lock_fast(a, ultimo=[], anchors=frozenset())
        except Exception:
            a = self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3)
        return sorted(set(a))


    def _reduzir_overlap_global(self, apostas: list[list[int]], limite: int, ultimo: list[int]) -> list[list[int]]:
        """
        Reduz greedy o overlap global: para cada par (i<j) com overlap>limite,
        altera apenas a aposta j usando trocas seguras (sem quebrar forma).
        Determinístico: varre pares na ordem natural.
        """
        if not apostas:
            return apostas
        universo = set(range(1, 26))
        out = [sorted(set(int(x) for x in a if 1 <= int(x) <= 25)) for a in apostas]

        def overlap(a, b) -> int:
            return len(set(a) & set(b))

        def _descolar(b: list[int], alvo: list[int]) -> list[int]:
            """Troca itens de b que colidem com 'alvo' por complementares, preservando forma."""
            b = sorted(set(b))
            comp = sorted(universo - set(b))
            colid = sorted((set(b) & set(alvo)), reverse=True)  # tira primeiro os mais altos (determinístico)
            for rem in colid:
                if len(set(b) & set(alvo)) <= limite:
                    break
                for add in comp:
                    if add in b:
                        continue
                    tmp = b[:]
                    # safe remove/add
                    try:
                        tmp.remove(rem)
                    except ValueError:
                        continue
                    tmp.append(add)
                    tmp = sorted(set(tmp))
                    pares = sum(1 for n in tmp if n % 2 == 0)
                    if (7 <= pares <= 8) and self._max_seq(tmp) <= 3:
                        b = tmp
                        comp = sorted(universo - set(b))
                        break
            # selagem curta
            try:
                b = self._hard_lock_fast(b, ultimo=ultimo or [], anchors=frozenset())
            except Exception:
                b = self._ajustar_paridade_e_seq(b, alvo_par=(7, 8), max_seq=3)
            return sorted(set(b))

        n = len(out)
        mudou = True
        # Limita iterações para evitar ciclos
        for _ in range(6):
            if not mudou:
                break
            mudou = False
            for i in range(n):
                for j in range(i + 1, n):
                    if overlap(out[i], out[j]) > limite:
                        novo = _descolar(out[j], out[i])
                        if novo != out[j]:
                            out[j] = novo
                            mudou = True
        return [sorted(a) for a in out]

    def _fechar_lote_stricto(self, apostas: list[list[int]], ultimo: list[int], overlap_max: int, max_ciclos: int = 6) -> list[list[int]]:
        """
        Fecha o lote até passar no TRIPLO CHECK (ou até max_ciclos).
          1) quebra seq>3 por aposta
          2) re-sela forma
          3) dedup
          4) reduz overlap global
          5) TRIPLO CHECK; se reprovar, repete
        """
        out = [sorted(set(int(x) for x in a if 1 <= int(x) <= 25)) for a in (apostas or [])]
        for _ in range(max_ciclos):
            # 1) quebrar seq>3 aposta a aposta
            out = [self._quebrar_seq_maior_3(a) for a in out]

            # 2) re-sela forma (curta)
            try:
                out = [self._hard_lock_fast(a, ultimo=ultimo or [], anchors=frozenset()) for a in out]
            except Exception:
                out = [self._ajustar_paridade_e_seq(a, alvo_par=(7, 8), max_seq=3) for a in out]

            # 3) dedup estrito
            uniq, seen = [], set()
            for a in out:
                t = tuple(a)
                if t not in seen:
                    seen.add(t)
                    uniq.append(a)
            out = uniq

            # 4) overlap global
            out = self._reduzir_overlap_global(out, overlap_max, ultimo or [])

            # 5) triplo check
            ok, _diag = self._triplo_check_stricto(out)
            if ok:
                break

        return [sorted(a) for a in out]

# ===================[ FIM UTILITÁRIOS STRICTO ]===================
        
    # --- Auditoria do lote + preparo do /refinar_bolão e dica do /mestre ---
    async def _auditar_e_preparar_refino(self, update, context, oficial_15: list[int]):
        """
        Usa o último lote registrado (learning.last_generation.apostas) e o oficial fornecido para:
          - TRIPLO CHECK-IN (regra inquebrável)
          - Contar acertos reais por aposta, média, melhor
          - Detectar coocorrências 'fortes'
          - Calcular reward/penalty por dezena (âncoras com escala reduzida)
          - Preparar bloco /refinar_bolão (α=0.36, janela=60) pronto para colar
          - Preparar bloco Mestre otimizado p/ próximo concurso com 20–30% de R-alto
        """
        # --- carrega o último lote persistido ---
        st = _normalize_state_defaults(_bolao_load_state() or {})
        learn = st.get("learning") or {}
        last_gen = learn.get("last_generation") or {}
        apostas = last_gen.get("apostas") or []

        if not apostas:
            return await update.message.reply_text(
                "Não encontrei um lote recente em memória (learning.last_generation). Gere um lote e tente novamente."
            )

        # --- TRIPLO CHECK (não interrompe mais o relatório) ---
        ok_lote, diag = self._triplo_check_stricto(apostas)  # retorna (bool, str) ou (bool, dict)

        # --- métricas sempre calculadas (mesmo com reprovação) ---
        oficial_set = set(oficial_15)

        def _hits_por_aposta(_aps: list[list[int]], _of: set[int]) -> list[int]:
            return [sum(1 for n in a if n in _of) for a in _aps]

        def _coocorrencias(_aps: list[list[int]]):
            from collections import Counter
            c = Counter()
            for a in _aps:
                s = sorted(a)
                for i in range(len(s)):
                    for j in range(i+1, len(s)):
                        c[(s[i], s[j])] += 1
            return c

        def _reward_penalty(_aps: list[list[int]], _of: set[int]):
            # âncoras recebem escala reduzida
            ANCHORS = set(globals().get("BOLAO_ANCHORS", (9, 11)))
            HIT = float(globals().get("BOLAO_BIAS_HIT", +0.5))
            MISS = float(globals().get("BOLAO_BIAS_MISS", -0.2))
            SCALE = float(globals().get("BOLAO_BIAS_ANCHOR_SCALE", 0.5))
            d = {n: 0.0 for n in range(1, 26)}
            for a in _aps:
                for n in a:
                    if n in _of:
                        d[n] += (HIT * (SCALE if n in ANCHORS else 1.0))
                    else:
                        d[n] += (MISS * (SCALE if n in ANCHORS else 1.0))
            return d

        def _format_dez(lst: list[int]) -> str:
            return " ".join(f"{n:02d}" for n in sorted(lst))

        acertos = _hits_por_aposta(apostas, oficial_set)
        media = sum(acertos) / float(len(acertos))
        melhor = max(acertos) if acertos else 0
        idx_melhores = [i+1 for i, v in enumerate(acertos) if v == melhor]

        cooc = _coocorrencias(apostas)
        top_pairs = sorted(cooc.items(), key=lambda kv: (-kv[1], kv[0]))[:10] if cooc else []

        deltas = _reward_penalty(apostas, oficial_set)
        recompensas = sorted([n for n, d in deltas.items() if d > 0], key=lambda n: (-deltas[n], n))
        penalizacoes = sorted([n for n, d in deltas.items() if d < 0], key=lambda n: (deltas[n], n))

        # ===== Bloco /refinar_bolão (α e janela preservados) =====
        alpha_fix = st.get("runtime", {}).get("alpha_usado", 0.36)
        janela_fix = int((st.get("learning") or {}).get("janela", 60))

        bias_update_lines = []
        for n in range(1, 26):
            delta = deltas.get(n, 0.0)
            if abs(delta) > 1e-9:
                bias_update_lines.append(f"{n:02d}:{delta:+.3f}")
        bias_blob = " ".join(bias_update_lines) if bias_update_lines else "(sem ajustes)"

        OVERLAP_MAX = int(globals().get("BOLAO_MAX_OVERLAP", 11))
        refinar_block = (
            f"/refinar_bolao alpha={alpha_fix:.2f} janela={janela_fix} "
            f"bias_delta=\"{bias_blob}\" "
            f"regras=\"paridade 7–8 | seq≤3 | anti-overlap≤{OVERLAP_MAX}\""
        )

        # ===== meta do Mestre (não mexe em estado se o lote reprovou) =====
        if ok_lote:
            learn_meta = learn.get("meta", {}) if isinstance(learn.get("meta", {}), dict) else {}
            learn_meta["R_alto_target"] = 0.25
            learn["meta"] = learn_meta
            st["learning"] = learn
            _bolao_save_state(st)

        mestre_hint = (
            "/mestre r_alto_target=0.25 "
            f"regras=\"paridade 7–8 | seq≤3 | anti-overlap≤{OVERLAP_MAX}\" ancoras=\"leves\""
        )

        # ===== Mensagem para o chat =====
        linhas = []

        if not ok_lote:
            linhas.append("⛔ <b>TRIPLO CHECK-IN FALHOU</b> — bloqueando aprendizado/reforço.\n")
            # diag pode ser str (relatório pronto) ou dict (estruturado)
            if isinstance(diag, dict):
                linhas.append("<b>🔎 TRIPLO CHECK (stricto)</b>")
                pf = diag.get("paridade_falhas") or []
                sf = diag.get("seq_falhas") or []
                ovf = diag.get("overlap_falhas") or []
                dups = diag.get("duplicatas") or []
                linhas.append(f"• Paridade: {'✅ todas em 7–8' if not pf else '❌ fora em '+str(pf)}")
                linhas.append(f"• Sequência: {'✅ OK' if not sf else '❌ >3 nas apostas: '+str(sf)}")
                if ovf:
                    worst = max(ovf, key=lambda t: t[2])
                    linhas.append(f"• Overlap máximo: ❌ {worst[2]} (> {OVERLAP_MAX}) entre Aposta {worst[0]:02d} e {worst[1]:02d}")
                else:
                    linhas.append(f"• Overlap máximo: ✅ ≤ {OVERLAP_MAX}")
                linhas.append(f"• Duplicatas: {'✅ nenhuma' if not dups else '❌ ' + str(dups)}")
                linhas.append("")
            else:
                linhas.append(str(diag) + "\n")

        # Sempre mostramos os resultados do lote (visibilidade total)
        linhas.append("🧮 <b>Acertos por aposta</b>")
        for i, ap in enumerate(apostas, 1):
            linhas.append(f"<b>Aposta {i:02d}:</b> {_format_dez(ap)}  →  <b>{acertos[i-1]}</b> acertos")
        linhas.append("")
        linhas.append("📊 <b>Resumo do Lote</b>")
        linhas.append(f"• Melhor aposta: <b>{melhor}</b> (IDs: {idx_melhores})")
        linhas.append(f"• Média do lote: <b>{media:.2f}</b> acertos")
        linhas.append(f"• Oficial: " + " ".join(f"{n:02d}" for n in sorted(oficial_15)))
        if ok_lote:
            linhas.append(f"• TRIPLO CHECK-IN: <b>OK</b> (paridade 7–8, seq≤3, anti-overlap≤{OVERLAP_MAX}, sem duplicatas)")
        else:
            linhas.append(f"• TRIPLO CHECK-IN: <b>REPROVADO</b> (sem reforço)")

        linhas.append("")
        linhas.append("🤝 <b>Coocorrências fortes</b> (top 10):")
        if top_pairs:
            linhas += [f"• ({a:02d},{b:02d}) → {c}x" for (a,b), c in top_pairs]
        else:
            linhas.append("• (n/d)")

        linhas.append("")
        linhas.append("⚖️ <b>Recompensas</b>: " + (_format_dez(recompensas) if recompensas else "(nenhuma)"))
        linhas.append("🔻 <b>Penalizações</b>: " + (_format_dez(penalizacoes) if penalizacoes else "(nenhuma)"))
        linhas.append("")

        # Só mostra o bloco de refino e a dica do Mestre se o triplo check passar
        if ok_lote:
            linhas.append("🛠️ <b>Bloco /refinar_bolão (copiar e colar):</b>")
            linhas.append(refinar_block)
            linhas.append("")
            linhas.append("🎯 <b>Próximo passo</b> (R-alto 20–30% habilitado):")
            linhas.append(mestre_hint)
        else:
            linhas.append("🔒 Lote reprovado: gere um novo lote conforme as regras e repita /confirmar.")

        # ✅ Envio longo (evita o erro “Message is too long”)
        texto = "\n".join(linhas)
        try:
            await self._send_long(update, texto, parse_mode="HTML")
        except Exception:
            logger.warning("Falha ao enviar com parse_mode HTML, tentando fallback...", exc_info=True)
            await self._send_long(update, texto, parse_mode=None)


    # ===== Registrar a última geração (para o aprendizado leve / auto_aprender) =====
    def _registrar_geracao(self, apostas: list[list[int]], base_resultado: list[int] | None):
        """
        Salva no estado persistente (bolao_state.json) a última geração entregue ao usuário.
        Isso é exatamente o que o auto_aprender usa depois para:
          - medir acertos médios,
          - ajustar bias por dezena,
          - recalibrar alpha, paridade e seq,
          - atualizar contadores de seen/hits.
        """
        from datetime import datetime
        from zoneinfo import ZoneInfo

        st = _normalize_state_defaults(_bolao_load_state() or {})
        st = dict(st) if isinstance(st, dict) else {}
        st.setdefault("learning", {})

        # tenta pegar snapshot atual; se falhar, usa "n/a"
        try:
            snap = self._latest_snapshot()
            snap_id = snap.snapshot_id
        except Exception:
            snap_id = "n/a"

        now_sp = datetime.now(ZoneInfo(TIMEZONE)).strftime("%Y-%m-%d %H:%M:%S %Z")

        st["learning"]["last_generation"] = {
            "timestamp": now_sp,
            "snapshot_id": snap_id,
            "base_resultado": [
                int(x) for x in sorted(int(n) for n in (base_resultado or []) if 1 <= int(n) <= 25)
            ],
            "apostas": [
                sorted(int(x) for x in ap if 1 <= int(x) <= 25)[:15]
                for ap in apostas
            ],
        }

        _bolao_save_state(st)

    # --- /estado_bolao: resumo do aprendizado atual (diagnóstico detalhado) ---
    async def estado_bolao(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Painel de auditoria do aprendizado leve do bolão/mestre.
        Entrega:
          • média de bias global e distribuição (+ / 0 / -)
          • top 5 dezenas mais favorecidas e mais punidas pelo bias
          • α (alpha) atual persistido
          • snapshot mais recente processado
          • timestamp do último registro de geração salva para aprendizado
          • últimas apostas registradas (A1..An)
          • overlap médio entre as apostas registradas
          • estimativa de ciclos (vezes que as dezenas já foram vistas)

        Objetivo: permitir avaliar se o auto_aprender está rodando de fato
        entre concursos e se o motor está ficando viciado demais em algumas dezenas.
        """
        try:
            # tenta carregar o estado persistido do bolão
            try:
                st = _normalize_state_defaults(_bolao_load_state() or {}) 
            except Exception:
                st = {}

            # bias bruto -> normalizar para dict[int] -> float
            raw_bias = st.get("bias", {}) or {}
            bias: dict[int, float] = {}
            for k, v in raw_bias.items():
                try:
                    bias[int(k)] = float(v)
                except Exception:
                    continue

            # cálculo estatístico do bias
            n_bias = len(bias)
            media_bias = (sum(bias.values()) / n_bias) if n_bias else 0.0
            positivos = [(dez, val) for dez, val in bias.items() if val > 0]
            negativos = [(dez, val) for dez, val in bias.items() if val < 0]
            neutros_count = n_bias - (len(positivos) + len(negativos))

            # top 5 mais puxados pra cima / mais penalizados
            top_pos = sorted(positivos, key=lambda kv: kv[1], reverse=True)[:5]
            top_neg = sorted(negativos, key=lambda kv: kv[1])[:5]

            # alpha atual
            try:
                alpha_eff = float(st.get("alpha", ALPHA_PADRAO))
            except Exception:
                alpha_eff = ALPHA_PADRAO

            # snapshot e info temporal
            snap_id = st.get("last_snapshot", st.get("snapshot", "--"))
            last_auto = st.get("last_auto", "--")

            # bloco learning salvo pelo _registrar_geracao()
            learning = st.get("learning", {}) or {}
            last_gen = learning.get("last_generation", {}) or {}
            ult_apostas = last_gen.get("apostas", [])
            ult_base = last_gen.get("base_resultado", [])
            ult_ts = last_gen.get("timestamp", "--")

            # estimativa de "ciclos": maior contador de aparição por dezena
            seen_map = st.get("seen", {}) or {}
            try:
                ciclos = max(int(v) for v in seen_map.values()) if seen_map else 0
            except Exception:
                ciclos = 0

            # calcular overlap médio entre as últimas apostas registradas
            try:
                overlaps = []
                for i in range(len(ult_apostas)):
                    for j in range(i + 1, len(ult_apostas)):
                        ai = set(int(x) for x in ult_apostas[i])
                        aj = set(int(x) for x in ult_apostas[j])
                        overlaps.append(len(ai & aj))
                med_overlap = (sum(overlaps) / len(overlaps)) if overlaps else 0.0
            except Exception:
                med_overlap = 0.0

            # formatador para pares dezena/bias
            def _fmt_pairs(pairs: list[tuple[int, float]]) -> str:
                if not pairs:
                    return "—"
                # exemplo: "02(+0.145)  17(+0.121)  20(+0.088)"
                return "  ".join(f"{dez:02d}(<i>{val:+.3f}</i>)" for dez, val in pairs)

            # montar resposta em HTML
            linhas: list[str] = []
            linhas.append("📈 <b>Estado do Aprendizado (Bolão / Mestre)</b>\n")

            linhas.append(
                f"• Média de bias global: <b>{media_bias:+.3f}</b> "
                f"(+{len(positivos)} | 0={neutros_count} | −{len(negativos)})"
            )
            linhas.append(f"• α atual (persistido): <b>{alpha_eff:.2f}</b>")
            linhas.append(f"• Snapshot mais recente aplicado: <b>{snap_id}</b>")
            linhas.append(f"• Último auto-aprendizado registrado: <b>{last_auto}</b>")
            linhas.append(f"• Timestamp da última geração salva: <b>{ult_ts}</b>")
            linhas.append(f"• Estimativa de ciclos (vistas acumuladas das dezenas): <b>{ciclos}</b>")

            # últimas apostas que serviram de base pro aprendizado
            if ult_apostas:
                linhas.append("\n🧠 <b>Última geração registrada (base do auto_aprender)</b>:")
                for i, ap in enumerate(ult_apostas, 1):
                    linhas.append(
                        f"- Aposta {i}: "
                        + " ".join(f"{n:02d}" for n in sorted(ap))
                    )
            else:
                linhas.append("\n🧠 <b>Última geração registrada (base do auto_aprender)</b>: (nenhuma)")

            # base_resultado associada
            if ult_base:
                linhas.append(
                    "• Base de referência (resultado observado no ciclo): "
                    + " ".join(f"{n:02d}" for n in sorted(ult_base))
                )

            # overlap
            linhas.append(
                f"\n• Overlap médio entre essas apostas: <b>{med_overlap:.2f}</b>"
            )
            linhas.append(
                f"• Limite alvo de overlap interno: ≤ <b>{BOLAO_MAX_OVERLAP}</b>"
            )

            # Top bias +
            linhas.append("\n<b>Top +5 (dezenas mais favorecidas pelo bias)</b>")
            linhas.append(_fmt_pairs(top_pos))

            # Top bias -
            linhas.append("\n<b>Top −5 (dezenas mais penalizadas pelo bias)</b>")
            linhas.append(_fmt_pairs(top_neg))

            return await update.message.reply_text(
                "\n".join(linhas),
                parse_mode="HTML",
            )

        except Exception:
            # fallback seguro em caso de qualquer erro inesperado
            try:
                st2 = _bolao_load_state() or {}
                raw_bias2 = st2.get("bias", {}) or {}
                vals = []
                for v in raw_bias2.values():
                    try:
                        vals.append(float(v))
                    except Exception:
                        pass
                media2 = (sum(vals) / len(vals)) if vals else 0.0
            except Exception:
                media2 = 0.0

            return await update.message.reply_text(
                f"📊 Média de bias atual: {media2:.3f}"
            )

    # --- /auto_aprender: rotina automática de aprendizado leve após cada concurso ---
    async def auto_aprender(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Aprendizado leve entre concursos / após fechamento de ciclo.
        Duas camadas com guard-rails:

        CAMADA 1 (micro por dezena):
        - Lê a última geração salva em st["learning"]["last_generation"]["apostas"].
        - Compara cada aposta com o último resultado oficial.
        - Para cada dezena (1..25):
            · Se a dezena apareceu nas apostas e saiu no resultado oficial → recompensa (BOLAO_BIAS_HIT).
            · Se a dezena apareceu e NÃO saiu → penaliza (BOLAO_BIAS_MISS).
          Âncoras usam escala reduzida (BOLAO_BIAS_ANCHOR_SCALE).
        - Atualiza mapas 'bias' (por dezena), 'hits' e 'seen'.

        CAMADA 2 (macro estrutural):
        - Mede a média real de acertos do lote (mu).
        - Mede paridade média (alvo 7–8), seq ≤ 3, e repetição (R).
        - Calcula propostas para meta-bias ('R','par','seq') e α.
        - **Política do usuário**: se média < 11 → RUIM → NÃO reforçar (nem α, nem bias).
          Se média ≥ 11 → pode reforçar, respeitando lock do α.

        Por fim, persiste com segurança e envia um resumo claro (sem "α agora:").
        """
        from statistics import mean
        from datetime import datetime
        from zoneinfo import ZoneInfo

        # ---- helpers locais (isolados) ----
        def _clamp_local(x: float, lo: float, hi: float) -> float:
            return lo if x < lo else hi if x > hi else x

        try:
            # ===== 0) Carregar estado atual e histórico =====
            st = _normalize_state_defaults(_bolao_load_state() or {})
            st = dict(st) if isinstance(st, dict) else {}
            st.setdefault("bias", {})
            st.setdefault("learning", {})
            st.setdefault("hits", {})
            st.setdefault("seen", {})
            st.setdefault("locks", {})
            st.setdefault("runtime", {})
            st.setdefault("policies", {})

            learning = st.get("learning", {}) or {}
            last_gen = learning.get("last_generation", {}) or {}
            ult_apostas: list[list[int]] = last_gen.get("apostas", []) or []

            if not ult_apostas:
                try:
                    await update.message.reply_text("🤖 Aprendizado leve: nenhuma geração registrada. (Nada a ajustar ainda.)")
                except Exception:
                    pass
                return

            historico = carregar_historico(HISTORY_PATH)
            if not historico:
                try:
                    await update.message.reply_text("🤖 Aprendizado leve: histórico indisponível.")
                except Exception:
                    pass
                return

            oficial = self._ultimo_resultado(historico)  # lista de 15 dezenas sorteadas (ordenada)
            oficial_set = set(int(x) for x in oficial)

            # ===== 1) CAMADA MICRO — ajuste de bias por dezena (com clamps locais) =====
            bias_num = {}
            for k, v in (st.get("bias") or {}).items():
                try:
                    kk = int(k)
                    if 1 <= kk <= 25:
                        bias_num[kk] = float(v)
                except Exception:
                    continue

            hits_map = st.get("hits", {}) or {}
            seen_map = st.get("seen", {}) or {}

            try:
                anchors_tuple = tuple(int(x) for x in BOLAO_ANCHORS)
            except Exception:
                anchors_tuple = tuple()
            anchors_set = set(anchors_tuple)

            for aposta in ult_apostas:
                aposta_set = set(int(x) for x in aposta)
                for dez in range(1, 26):
                    # contabiliza "tentativa" de uso
                    seen_map[str(dez)] = int(seen_map.get(str(dez), 0)) + (1 if dez in aposta_set else 0)

                    if dez not in aposta_set:
                        continue

                    # recompensa/penalidade
                    if dez in oficial_set:
                        scale = BOLAO_BIAS_ANCHOR_SCALE if dez in anchors_set else 1.0
                        delta = float(BOLAO_BIAS_HIT) * scale
                        hits_map[str(dez)] = int(hits_map.get(str(dez), 0)) + 1
                    else:
                        scale = BOLAO_BIAS_ANCHOR_SCALE if dez in anchors_set else 1.0
                        delta = float(BOLAO_BIAS_MISS) * scale

                    # aplica ajuste com clamp local
                    bias_num[dez] = _clamp_local(
                        float(bias_num.get(dez, 0.0)) + float(delta),
                        float(BOLAO_BIAS_MIN),
                        float(BOLAO_BIAS_MAX),
                    )

            # snapshot_id atual (não crítico, mas útil para telemetria)
            try:
                snap = self._latest_snapshot()
                st["last_snapshot"] = getattr(snap, "snapshot_id", "--")
            except Exception:
                st["last_snapshot"] = "--"

            # ===== 2) CAMADA MACRO — métricas e propostas de ajuste =====
            hits_list = [self._contar_acertos(ap, oficial) for ap in ult_apostas]
            mu_hits = mean(hits_list) if hits_list else 0.0  # média REAL (R)

            seq_list = [self._max_seq(ap) for ap in ult_apostas]
            seq_mu = mean(seq_list) if seq_list else 0.0
            seq_viol = sum(1 for s in seq_list if s > 3)

            pares_medios = mean(self._paridade(ap)[0] for ap in ult_apostas) if ult_apostas else 0.0

            # alvo de repetição médio (9R–10R ~ 9.5)
            alvo_R = 9.5
            delta_R = mu_hits - alvo_R

            # meta-bias globais (R, paridade, seq) — valores atuais
            bias_global_R   = float((st.get("bias") or {}).get("R", 0.0))
            bias_global_par = float((st.get("bias") or {}).get("paridade", 0.0))
            bias_global_seq = float((st.get("bias") or {}).get("seq", 0.0))

            # ganhos pequenos e estáveis
            k_R, k_P, k_S = 0.02, 0.01, 0.02

            prop_bias_R   = _clamp_local(bias_global_R   - k_R * delta_R,                           -0.20,  0.20)
            prop_bias_par = _clamp_local(bias_global_par + k_P * (7.5 - pares_medios),              -0.15,  0.15)
            prop_bias_seq = _clamp_local(bias_global_seq + k_S * ((seq_mu - 3.0) + 0.5 * (seq_viol / max(1, len(ult_apostas)))), -0.20, 0.20)

            # proposta de alpha (α) baseada na média — conservadora (NÃO aplicada se lock ativo ou se média < 11)
            alpha_atual = float(st.get("alpha", ALPHA_PADRAO))
            if   mu_hits < 9.0:
                delta_alpha = -0.01  # média baixa → abrir diversidade
            elif mu_hits > 10.0:
                delta_alpha = 0.01   # média alta  → concentrar repetição
            else:
                delta_alpha = 0.0

            alpha_proposto_novo = _clamp_local(alpha_atual + delta_alpha, 0.30, 0.42)

            # ===== 3) Política do usuário (corte ≥ 11) + Locks/Gates + Persistência =====
            # classifica qualidade (usa sua função utilitária)
            qualidade = self._classificar_lote_por_media(float(mu_hits))  # "RUIM" se < 11
            aplicar_reforco = (qualidade != "RUIM")

            # locks/gates
            lock_ativo = bool(st["locks"].get("alpha_travado", True))
            official_gate = bool(st["policies"].get("official_gate", True))

            # monta dicionário final de bias (numérico por dezena + globais), mas só aplica se reforço permitido
            new_bias_out: dict[str, float] = {}
            for dez, val in bias_num.items():
                new_bias_out[str(dez)] = float(val)
            new_bias_out["R"]        = float(prop_bias_R)
            new_bias_out["paridade"] = float(prop_bias_par)
            new_bias_out["seq"]      = float(prop_bias_seq)

            # aplica ou não aplica reforço
            if aplicar_reforco:
                # salva meta-bias (agregados) também no learning (telemetria)
                st["bias"] = new_bias_out
                learning.setdefault("bias_meta", {})
                learning["bias_meta"] = {
                    "R":   float(prop_bias_R),
                    "par": float(_clamp_local(prop_bias_par, -0.10, 0.10)),  # janela um pouco mais estreita p/ paridade
                    "seq": float(prop_bias_seq),
                }

                # α: respeita lock/gate
                if lock_ativo:
                    learning["alpha_proposto"] = float(alpha_proposto_novo)
                else:
                    if official_gate:
                        learning["alpha"] = float(alpha_proposto_novo)
                        st["runtime"]["alpha_usado"] = float(alpha_proposto_novo)
                        st["alpha"] = float(alpha_proposto_novo)
                    else:
                        learning["alpha_proposto"] = float(alpha_proposto_novo)

                msg_bias = (
                    f"bias[R]={learning['bias_meta'].get('R', 0.0):+.3f}  "
                    f"bias[par]={learning['bias_meta'].get('par', 0.0):+.3f}  "
                    f"bias[seq]={learning['bias_meta'].get('seq', 0.0):+.3f}"
                )
                if lock_ativo:
                    msg_alpha = f"α proposto: {float(learning.get('alpha_proposto', alpha_proposto_novo)):.2f} (pendente; lock ativo)"
                else:
                    msg_alpha = f"α usado: {float(learning.get('alpha', st['runtime'].get('alpha_usado', alpha_atual))):.2f} (livre)"
            else:
                # NÃO reforça nada em lote RUIM (<11)
                # mantém st["bias"] como estava; learning.bias_meta permanece como estava (se existir)
                msg_bias  = "bias inalterado (lote RUIM)"
                msg_alpha = f"α mantido: {float(st['runtime'].get('alpha_usado', ALPHA_LOCK_VALUE)):.2f} (lote RUIM; sem reforço)"

            # campos auxiliares persistidos
            st["hits"] = hits_map
            st["seen"] = seen_map
            learning["last_learn"] = {
                "updated_at": datetime.now(ZoneInfo(TIMEZONE)).isoformat(),
                "media_real": round(float(mu_hits), 4),
                "janela": int(learning.get("janela", JANELA_PADRAO)),
                "diag": {"ok": aplicar_reforco, "qualidade": qualidade},
            }
            st["learning"] = learning
            st["last_auto"] = datetime.now(ZoneInfo(TIMEZONE)).strftime("%Y-%m-%d %H:%M:%S %Z")

            try:
                _bolao_save_state(st)
            except Exception:
                pass

            # ===== 4) Feedback resumido no chat (revisado; sem "α agora:") =====
            try:
                st_safe    = _normalize_state_defaults(_bolao_load_state() or {})
                runtime    = st_safe.get("runtime") or {}
                locks      = st_safe.get("locks") or {}
                learn_safe = st_safe.get("learning") or {}

                alpha_usado_msg = float(runtime.get("alpha_usado", ALPHA_LOCK_VALUE))
                lock_view = bool(locks.get("alpha_travado", True))
                alpha_prop_view = learn_safe.get("alpha_proposto", None)
                bias_meta_view  = learn_safe.get("bias_meta", {}) or {}

                alpha_info = f"α usado: {alpha_usado_msg:.2f} (travado)" if lock_view else f"α usado: {alpha_usado_msg:.2f} (livre)"
                if qualidade == "RUIM":
                    msg_alpha_view = msg_alpha  # já montado acima
                    msg_bias_view  = msg_bias
                else:
                    if lock_view:
                        msg_alpha_view = f"α proposto: {float(alpha_prop_view):.2f} (pendente; lock ativo)" if alpha_prop_view is not None else "α sem proposta pendente (lock ativo)"
                    else:
                        alpha_oficial = float(learn_safe.get("alpha", alpha_usado_msg))
                        msg_alpha_view = f"α usado: {alpha_oficial:.2f} (livre)"
                    msg_bias_view = (
                        f"bias[R]={bias_meta_view.get('R', 0.0):+.3f}  "
                        f"bias[par]={bias_meta_view.get('par', 0.0):+.3f}  "
                        f"bias[seq]={bias_meta_view.get('seq', 0.0):+.3f}"
                    )

                await update.message.reply_text(
                    "📈 Aprendizado leve atualizado.\n"
                    f"• Lote avaliado: {len(ult_apostas)} apostas\n"
                    f"• Média de acertos: {float(mu_hits):.2f}  →  Qualidade: {qualidade} (alvo ≥ {DESEMPENHO_MINIMO_R:.0f})\n"
                    f"• {alpha_info}\n"
                    f"• {msg_alpha_view}\n"
                    f"• {msg_bias_view}"
                )
            except Exception:
                # se der erro só no reply_text não queremos quebrar o fluxo
                pass

        except Exception:
            logger.warning("auto_aprender falhou internamente.", exc_info=True)
    

    @staticmethod
    def _contar_repeticoes(aposta, ultimo):
        u = set(ultimo)
        return sum(1 for n in aposta if n in u)

    # --- Novo comando: /mestre ---
    async def mestre(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        import asyncio, traceback
        from datetime import datetime
        from zoneinfo import ZoneInfo

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
            ultimo = ultimo_sorted if ultimo_sorted else []
        except Exception:
            ultimo = []
        try:
            self._registrar_geracao(apostas, base_resultado=ultimo)
        except Exception:
            logger.warning("Falha ao registrar geração para aprendizado leve (/mestre).", exc_info=True)

        # >>> NOVO: registrar o lote no estado (pending_batches)
        try:
            st = _normalize_state_defaults(_bolao_load_state() or {})
            batches = st.get("pending_batches", [])

            batches.append({
                "ts": datetime.now(ZoneInfo(TIMEZONE)).isoformat(),
                "snapshot": getattr(self._latest_snapshot(), "snapshot_id", "--"),
                "alpha": float(st.get("alpha", ALPHA_PADRAO)),
                "janela": int((st.get("learning") or {}).get("janela", JANELA_PADRAO)),
                "oficial_base": " ".join(f"{n:02d}" for n in (ultimo or [])),
                "qtd": len(apostas),
                # opcional: salvar as apostas (atenção ao tamanho do estado)
                "apostas": [" ".join(f"{x:02d}" for x in a) for a in apostas],
            })

            # mantém histórico curto de lotes pendentes
            st["pending_batches"] = batches[-100:]
            _bolao_save_state(st)

        except Exception:
            logger.warning("Falha ao registrar pending_batch (/mestre).", exc_info=True)
        # <<< FIM NOVO

        # --- Telemetria e formatação da resposta ---
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

        linhas_str = "\n".join(linhas)

        # envia pro usuário
        await update.message.reply_text(linhas_str, parse_mode="HTML")

        # --- Aprendizado automático pós-envio (com gating ativo ele retorna sem mexer) ---
        try:
            await self.auto_aprender(update, context)
        except Exception:
            logger.warning("auto_aprender falhou pós-/mestre; prosseguindo normalmente.", exc_info=True)

    async def confirmar(self, update, context):
        try:
            nums = [int(x) for x in (update.message.text.split()[1:])]
            if len(nums) != 15 or any(n < 1 or n > 25 for n in nums):
                return await update.message.reply_text("Use: /confirmar <15 dezenas entre 1..25>")
            nums = sorted(nums)
            await self._auditar_e_preparar_refino(update, context, nums)
        except Exception as e:
            logger.error("Erro no /confirmar:\n" + traceback.format_exc())
            return await update.message.reply_text(f"Erro no /confirmar: {e}")

    # --- /mestre_bolao: Modo Bolão v5 (19→15) selado e estável, com timeout seguro ---
    async def mestre_bolao(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Gera lote 19→15 a partir de uma matriz-19 estável do histórico,
        preservando âncoras, com selagem forte:
          - Paridade 7–8
          - SeqMax ≤ 3
          - Dedup + anti-overlap ≤ BOLAO_MAX_OVERLAP
        Inclui proteção de timeout na geração 19→15 com fallback determinístico.
        """
        import asyncio, traceback
        from datetime import datetime
        from zoneinfo import ZoneInfo

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

        chat_id = update.message.chat_id if update.message else update.effective_chat.id
        if self._hit_cooldown(chat_id, "mestre_bolao"):
            return await update.message.reply_text(f"⏳ Aguarde {COOLDOWN_SECONDS}s para usar /mestre_bolao novamente.")

        try:
            # --- carregar histórico e último resultado ---
            historico = carregar_historico(HISTORY_PATH)
            if not historico:
                return await update.message.reply_text("Erro: histórico vazio.")
            ultimo = self._ultimo_resultado(historico)

            # --- matriz-19 estável usando sua rotina oficial ---
            try:
                matriz19 = self._selecionar_matriz19(historico)
            except Exception:
                # fallback: último + âncoras + completa até 19
                universo = list(range(1, 26))
                base = sorted({n for n in (ultimo or []) if 1 <= n <= 25})
                for a in BOLAO_ANCHORS:
                    if a not in base:
                        base.append(a)
                for n in universo:
                    if len(base) >= 19:
                        break
                    if n not in base:
                        base.append(n)
                matriz19 = sorted(base)[:19]

            # --- seed determinística por snapshot/usuário ---
            try:
                snap   = self._latest_snapshot()
                s_inc  = self._next_draw_seed(snap.snapshot_id)
            except Exception:
                snap, s_inc = None, self._next_draw_seed("fallback")
            user_seed = self._calc_mestre_seed(
                user_id=user_id,
                chat_id=chat_id,
                ultimo_sorted=sorted(ultimo),
            )
            seed = (int(s_inc) ^ (int(user_seed) & 0xFFFFFFFF)) & 0xFFFFFFFF

            # --- gerar 19→15 com timeout + fallback seguro ---
            async def _run_expand():
                return await asyncio.to_thread(self._subsets_19_para_15, matriz19, seed)

            try:
                apostas = await asyncio.wait_for(_run_expand(), timeout=8.0)
            except asyncio.TimeoutError:
                logger.warning("/mestre_bolao: expansão 19→15 >8s; usando fallback determinístico.")
                # fallback linear a partir da matriz19
                base = sorted(matriz19)
                apostas = []
                for off in range(min(BOLAO_QTD_APOSTAS, 10)):
                    s = []
                    idx = off
                    while len(s) < 15:
                        s.append(base[idx % len(base)])
                        idx += 1
                    apostas.append(sorted(set(s)))
            except Exception:
                logger.error("Erro na expansão 19→15:\n" + traceback.format_exc())
                return await update.message.reply_text("Erro ao expandir 19→15. Tente novamente.")

            # --- selagem por aposta + pós-filtro unificado ---
            try:
                anchors = frozenset(BOLAO_ANCHORS)
            except Exception:
                anchors = frozenset()
            apostas = [self._hard_lock_fast(a, ultimo=ultimo, anchors=anchors) for a in apostas]
            try:
                apostas = self._pos_filtro_unificado(apostas, ultimo=ultimo)
            except Exception:
                logger.warning("Pós-filtro unificado falhou no /mestre_bolao; mantendo selagem rápida.", exc_info=True)

            # --- validação final teimosa ---
            apostas_ok = []
            for a in apostas[:BOLAO_QTD_APOSTAS]:
                a = self._hard_lock_fast(a, ultimo=ultimo, anchors=anchors)
                apostas_ok.append(a)
            apostas = apostas_ok

            # --- REGISTRO p/ aprendizado leve ---
            try:
                self._registrar_geracao(apostas, base_resultado=ultimo or [])
            except Exception:
                logger.warning("Falha ao registrar geração para aprendizado leve (/mestre_bolao).", exc_info=True)

            # >>> NOVO: registrar o lote no estado (pending_batches)
            try:
                st = _normalize_state_defaults(_bolao_load_state() or {})
                batches = st.get("pending_batches", [])
                batches.append({
                    "ts": datetime.now(ZoneInfo(TIMEZONE)).isoformat(),
                    "snapshot": getattr(self._latest_snapshot(), "snapshot_id", "--"),
                    "alpha": float(st.get("alpha", ALPHA_PADRAO)),
                    "janela": int((st.get("learning") or {}).get("janela", JANELA_PADRAO)),
                    "oficial_base": " ".join(f"{n:02d}" for n in (ultimo or [])),
                    "qtd": len(apostas),
                    # opcional: salvar as apostas (cuidado com o tamanho do estado)
                    "apostas": [" ".join(f"{x:02d}" for x in a) for a in apostas],
                })
                st["pending_batches"] = batches[-100:]
                _bolao_save_state(st)
            except Exception:
                logger.warning("Falha ao registrar pending_batch no /mestre_bolao.", exc_info=True)
            # <<< FIM NOVO

            # >>> CSV de auditoria (aprendizado REAL já existente aqui)
            try:
                snap = self._latest_snapshot()
                _append_learn_log(snap.snapshot_id, ultimo or [], apostas)
            except Exception:
                logger.warning("Falha para append no learn_log após /mestre_bolao.", exc_info=True)

            # --- resposta formatada (telemetria resumida) ---
            linhas = ["🎰 <b>SUAS APOSTAS INTELIGENTES — Modo Bolão v5</b> 🎰\n"]
            ok_count = 0
            u_set = set(ultimo)
            for i, a in enumerate(apostas, 1):
                pares = self._contar_pares(a)
                seq   = self._max_seq(a)
                rep   = sum(1 for n in a if n in u_set)
                ok    = (7 <= pares <= 8) and (seq <= 3)
                if ok:
                    ok_count += 1
                linhas.append(
                    f"<b>Aposta {i}:</b> {' '.join(f'{n:02d}' for n in a)}\n"
                    f"🔢 Pares: {pares} | Ímpares: {15 - pares} | SeqMax: {seq} | {rep}R | {'✅ OK' if ok else '🛠️'}\n"
                )

            linhas.append(f"\n<b>Conformidade</b>: {ok_count}/{len(apostas)} dentro de (paridade 7–8, seq≤3)")
            linhas.append(f"<i>Regras: paridade 7–8, seq≤3, anti-overlap≤{BOLAO_MAX_OVERLAP}</i>")

            if SHOW_TIMESTAMP:
                from hashlib import md5
                try:
                    hash_ult = md5("".join(f"{n:02d}" for n in sorted(ultimo)).encode()).hexdigest()[:8]
                except Exception:
                    hash_ult = "--"
                snap_id = getattr(snap, "snapshot_id", "n/a")
                carimbo = datetime.now(ZoneInfo(TIMEZONE)).strftime("%Y-%m-%d %H:%M:%S %Z")
                linhas.append(f"<i>base=último | hash={hash_ult} | snapshot={snap_id} | {carimbo}</i>")

            linhas_str = "\n".join(linhas)

            # envia pro usuário
            await update.message.reply_text(linhas_str, parse_mode="HTML")

            # aprendizado pós-envio (não travar bot se falhar)
            try:
                await self.auto_aprender(update, context)
            except Exception:
                logger.warning("auto_aprender falhou pós-/mestre_bolao; prosseguindo normalmente.", exc_info=True)

            return

        except Exception as e:
            logger.error("Erro no /mestre_bolao:\n" + traceback.format_exc())
            return await update.message.reply_text(f"Erro no /mestre_bolao: {e}")

    # --- ping / versao / diagbase já definidos acima ---
    async def ping(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("pong")

    async def versao(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        txt = (
            f"🤖 Versão do bot\n"
            f"- BUILD_TAG: <code>{BUILD_TAG}</code>\n"
            f"- Import layout: <code>{LAYOUT}</code>\n"
            f"- Comandos: /start /gerar /mestre /mestre_bolao /refinar_bolao /meuid /autorizar /remover /backtest /diagbase /ping /versao"
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

    # (REMOVIDO) Comando /ab e variações (incluindo "/ab c" ou "/ab ciclo")
    # Esta função foi removida por decisão de produto. Caso receba alguma chamada  
    # acidental, a camada de unknown command responderá de forma neutra.
    # Mantemos apenas este comentário de sentinela para facilitar auditorias futuras.

    # ------------- Handler do backtest -------------
    async def backtest(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Executa um backtest offline (somente admin).
        NÃO alimenta aprendizado leve nem registra geração,
        porque isso iria corromper o estado atual com dados históricos antigos.
        """
        import asyncio, traceback
        from functools import partial

        user_id = update.effective_user.id
        if not self._is_admin(user_id):
            return  # só admin pode rodar backtest

        # (anti-abuso redundante para admin é desnecessário, mas não faz mal.
        # Mantemos só um aviso de execução.)
        janela, bilhetes_por_concurso, alpha = self._parse_backtest_args(context.args)

        await update.message.reply_text(
            f"Executando backtest com janela={janela}, "
            f"bilhetes={bilhetes_por_concurso}, "
            f"α={alpha:.2f}."
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

    # --- Execução principal do bot ---
    def run(self):
        """
        Sobe o bot em modo polling.
        MUITO IMPORTANTE: rode apenas UMA instância desse processo.
        Se você abrir 2 processos rodando .run_polling() com o MESMO token,
        o Telegram responde com "Conflict: terminated by other getUpdates request".
        """
        logger.info("Bot iniciado e aguardando comandos.")
        # run_polling() já cuida do loop interno asyncio do python-telegram-bot 20.x
        self.app.run_polling(
            allowed_updates=Update.ALL_TYPES,
            stop_signals=None,  # mantemos processo principal responsável por encerrar
            close_loop=False,
        )

# ========== ENTRYPOINT ==========
if __name__ == "__main__":
    try:
        bot = LotoFacilBot()
        bot.run()
    except Exception:
        logger.error("Falha fatal ao iniciar o bot:\n" + traceback.format_exc())
