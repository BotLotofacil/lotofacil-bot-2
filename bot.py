# bot.py

import os
import logging
import traceback
import asyncio
import re
from functools import partial
from typing import List, Set, Tuple
from datetime import datetime
from zoneinfo import ZoneInfo

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from dotenv import load_dotenv

# --- Imports do gerador preditivo ---
from utils.history import carregar_historico, ultimos_n_concursos
from utils.predictor import Predictor, GeradorApostasConfig
# --- Import do backtest ---
from utils.backtest import executar_backtest_resumido

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
# Parâmetros padrão do gerador
# ========================
# Definições padrão fixas (sem necessidade de passar argumentos no /gerar):
#   - quantidade: 5 apostas
#   - janela:     100 concursos do histórico
#   - alpha:      0.45 (equilíbrio estatística vs uniforme)
JANELA_PADRAO = 100
ALPHA_PADRAO = 0.45
QTD_BILHETES_PADRAO = 5

SHOW_TIMESTAMP = True
TIMEZONE = "America/Sao_Paulo"

# Limites defensivos
JANELA_MIN, JANELA_MAX = 50, 1000
ALPHA_MIN, ALPHA_MAX = 0.05, 0.95
BILH_MIN, BILH_MAX   = 1, 20

HISTORY_PATH = "data/history.csv"
WHITELIST_PATH = "whitelist.txt"

# ========================
# Bot Principal
# ========================
class LotoFacilBot:
    def __init__(self):
        self.token = self._get_bot_token()
        self.admin_id = self._get_admin_id()
        self.whitelist_path = WHITELIST_PATH
        self.whitelist = self._carregar_whitelist()
        self._garantir_admin_na_whitelist()
        self.app = ApplicationBuilder().token(self.token).build()
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
        Em caso de falha, aplica fallback uniforme.
        """
        try:
            qtd, janela, alpha = self._clamp_params(qtd, janela, alpha)
            historico = carregar_historico(HISTORY_PATH)
            janela_hist = ultimos_n_concursos(historico, janela)

            cfg = GeradorApostasConfig(janela=janela, alpha=alpha)
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
          - Chave=valor: /backtest janela=200 bilhetes=5 alpha=0,45
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

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /start – mensagem de boas-vindas e aviso legal."""
        mensagem = (
            "⚠️ <b>Aviso Legal</b>\n"
            "Este bot é apenas para fins estatísticos e recreativos. "
            "Não há garantia de ganhos na Lotofácil.\n\n"
            "🎉 <b>Bem-vindo</b>\n"
            "Use /gerar para receber 5 apostas baseadas em 100 concursos e α=0.45.\n"
            "Use /meuid para obter seu identificador e solicitar autorização.\n"
        )
        await update.message.reply_text(mensagem, parse_mode="HTML")

    async def gerar_apostas(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Comando /gerar – Gera apostas inteligentes.
        Uso: /gerar [qtd] [janela] [alpha]
        Padrão: 5 apostas | janela=100 | α=0.45
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
        Padrão: janela=100 | bilhetes=5 | α=0.45
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

