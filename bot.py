# bot.py

import os
import logging
import traceback
import asyncio
import re
from functools import partial
from typing import List, Set, Tuple, Optional
from datetime import datetime
from zoneinfo import ZoneInfo

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from dotenv import load_dotenv

# --- Imports do gerador preditivo ---
from utils.history import carregar_historico, ultimos_n_concursos
from utils.predictor import Predictor, GeradorApostasConfig
# --- Import do backtest (novo) ---
from utils.backtest import executar_backtest_resumido

# ========================
# Carrega variáveis de ambiente locais
# ========================
load_dotenv()

# ========================
# Logging
# ========================
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ========================
# Parâmetros padrão do gerador
# ========================
JANELA_PADRAO = 50     # concursos usados no treino
ALPHA_PADRAO = 0.55     # mistura uniforme vs estimado
QTD_BILHETES_PADRAO = 3 # quantidade padrão de apostas por /gerar

SHOW_TIMESTAMP = True                 # False para ocultar a hora no /gerar
TIMEZONE = "America/Sao_Paulo"        # Fuso para exibição do horário

# Limites defensivos
JANELA_MIN, JANELA_MAX = 50, 1000
ALPHA_MIN, ALPHA_MAX = 0.05, 0.95
BILH_MIN, BILH_MAX   = 1, 20

# ========================
# Bot Principal
# ========================
class LotoFacilBot:
    def __init__(self):
        self.token = self._get_bot_token()
        self.admin_id = self._get_admin_id()
        self.whitelist_path = "whitelist.txt"
        self.whitelist = self._carregar_whitelist()
        self._garantir_admin_na_whitelist()
        self.app = ApplicationBuilder().token(self.token).build()
        self._setup_handlers()

    # ------------- Utilidades internas -------------
    def _get_bot_token(self) -> str:
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            raise EnvironmentError("❌ Variável TELEGRAM_BOT_TOKEN não configurada.")
        return token

    def _get_admin_id(self) -> int:
        admin_id = os.getenv("ADMIN_TELEGRAM_ID")
        if not admin_id or not admin_id.isdigit():
            raise EnvironmentError("❌ ADMIN_TELEGRAM_ID não configurado corretamente.")
        return int(admin_id)

    def _carregar_whitelist(self) -> Set[int]:
        """Carrega os IDs autorizados do arquivo de whitelist"""
        if not os.path.exists(self.whitelist_path):
            return set()
        with open(self.whitelist_path, "r") as f:
            return set(int(l.strip()) for l in f if l.strip().isdigit())

    def _salvar_whitelist(self):
        """Salva a whitelist atual no arquivo"""
        with open(self.whitelist_path, "w") as f:
            for user_id in sorted(self.whitelist):
                f.write(f"{user_id}\n")

    def _garantir_admin_na_whitelist(self):
        """Garante que o administrador esteja sempre autorizado"""
        if self.admin_id not in self.whitelist:
            self.whitelist.add(self.admin_id)
            self._salvar_whitelist()
            logging.info(f"✅ Administrador {self.admin_id} autorizado automaticamente.")

    def _usuario_autorizado(self, user_id: int) -> bool:
        return user_id in self.whitelist

    def _is_admin(self, user_id: int) -> bool:
        return user_id == self.admin_id

    # ------------- Gerador preditivo -------------
    def _gerar_apostas_inteligentes(
        self,
        qtd: int = QTD_BILHETES_PADRAO,
        janela: int = JANELA_PADRAO,
        alpha: float = ALPHA_PADRAO
    ) -> List[List[int]]:
        """
        Gera bilhetes usando o preditor sem enviesar.
        Em caso de falha na leitura/treino, faz fallback para amostragem uniforme.
        """
        try:
            historico = carregar_historico("data/history.csv")
            janela_hist = ultimos_n_concursos(historico, max(30, int(janela)))

            cfg = GeradorApostasConfig(janela=int(janela), alpha=float(alpha))
            modelo = Predictor(cfg)
            modelo.fit(janela_hist, janela=int(janela))
            bilhetes = modelo.gerar_apostas(qtd=int(qtd))
            return bilhetes
        except Exception:
            logger.error("Falha no gerador preditivo; aplicando fallback.\n" + traceback.format_exc())
            import random
            rng = random.Random()
            qtd_seguro = max(1, int(qtd))
            return [sorted(rng.sample(range(1, 26), 15)) for _ in range(qtd_seguro)]

    # ------------- Parse utilitário p/ backtest -------------
    def _parse_backtest_args(self, args: List[str]) -> Tuple[int, int, float]:
        """
        Aceita:
          - Posicional: /backtest [janela] [bilhetes_por_concurso] [alpha]
          - Chave=valor: /backtest janela=200 bilhetes=5 alpha=0,45
          - Aliases: j=, b=, a= funcionam
        Retorna tupla validada (janela, bilhetes_por_concurso, alpha).
        Aplica defaults/limites quando ausentes/fora de faixa.
        """
        # Defaults
        janela = JANELA_PADRAO
        bilhetes_por_concurso = 5
        alpha = ALPHA_PADRAO

        if not args:
            return janela, bilhetes_por_concurso, alpha

        # Normalização
        joined = " ".join(args).strip()
        joined = joined.replace(",", ".")
        joined = re.sub(r"\bj\s*=", "janela=", joined)
        joined = re.sub(r"\bb(ilhetes)?\s*=", "bilhetes=", joined)
        joined = re.sub(r"\ba\s*=", "alpha=", joined)

        # Se houver chave=valor, prioriza esse modo
        has_kv = bool(re.search(r"\b(janela|bilhetes|alpha)\s*=", joined))

        if has_kv:
            m_j = re.search(r"\bjanela\s*=\s*(\d{1,5})\b", joined)
            if m_j:
                try:
                    j_val = int(m_j.group(1))
                    if JANELA_MIN <= j_val <= JANELA_MAX:
                        janela = j_val
                except ValueError:
                    pass

            m_b = re.search(r"\bbilhetes\s*=\s*(\d{1,3})\b", joined)
            if m_b:
                try:
                    b_val = int(m_b.group(1))
                    if BILH_MIN <= b_val <= BILH_MAX:
                        bilhetes_por_concurso = b_val
                except ValueError:
                    pass

            m_a = re.search(r"\balpha\s*=\s*([01]?(?:\.\d+)?)\b", joined)
            if m_a:
                try:
                    a_val = float(m_a.group(1))
                    if ALPHA_MIN <= a_val <= ALPHA_MAX:
                        alpha = a_val
                except ValueError:
                    pass
        else:
            # Posicional
            try:
                if len(args) >= 1:
                    j_val = int(args[0])
                    if JANELA_MIN <= j_val <= JANELA_MAX:
                        janela = j_val
                if len(args) >= 2:
                    b_val = int(args[1])
                    if BILH_MIN <= b_val <= BILH_MAX:
                        bilhetes_por_concurso = b_val
                if len(args) >= 3:
                    a_val = float(args[2].replace(",", "."))
                    if ALPHA_MIN <= a_val <= ALPHA_MAX:
                        alpha = a_val
            except Exception:
                # Ignora erro e mantém defaults já validados
                pass

        return janela, bilhetes_por_concurso, alpha

    # ------------- Handlers -------------
    def _setup_handlers(self):
        """Registra os comandos do bot"""
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("gerar", self.gerar_apostas))
        self.app.add_handler(CommandHandler("meuid", self.meuid))
        self.app.add_handler(CommandHandler("autorizar", self.autorizar))
        self.app.add_handler(CommandHandler("remover", self.remover))
        # --- Handler do backtest (novo) ---
        self.app.add_handler(CommandHandler("backtest", self.backtest))

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /start – Mensagem de boas-vindas e aviso legal"""
        mensagem = (
            "⚠️ <b>Aviso Legal</b>\n"
            "Este bot é apenas para fins estatísticos e recreativos. "
            "Não há qualquer garantia de ganhos na Lotofácil ou em qualquer loteria.\n\n"
            "🎉 <b>Bem-vindo ao Bot de Apostas Inteligentes da Lotofácil</b>!\n"
            "Use /gerar para receber apostas baseadas em estratégias avançadas.\n"
            "Use /meuid para obter seu identificador e solicitar autorização.\n"
        )
        await update.message.reply_text(mensagem, parse_mode='HTML')

    async def gerar_apostas(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Comando /gerar – Gera apostas inteligentes (com verificação de autorização).
        Uso opcional: /gerar [qtd] [janela] [alpha]
        """
        user_id = update.effective_user.id
        logger.info(f"Comando /gerar chamado por {user_id}")

        if not self._usuario_autorizado(user_id):
            await update.message.reply_text("⛔ Você não está autorizado a gerar apostas.")
            return

        # parâmetros padrão
        qtd = QTD_BILHETES_PADRAO
        janela = JANELA_PADRAO
        alpha = ALPHA_PADRAO

        # leitura opcional de argumentos
        try:
            if context.args and len(context.args) >= 1:
                qtd = max(1, int(context.args[0]))
            if context.args and len(context.args) >= 2:
                janela = max(30, int(context.args[1]))  # mínimo razoável
            if context.args and len(context.args) >= 3:
                a = float(context.args[2].replace(",", "."))
                alpha = a if 0.0 < a < 1.0 else ALPHA_PADRAO
        except Exception:
            # argumentos inválidos -> usa padrão
            qtd, janela, alpha = QTD_BILHETES_PADRAO, JANELA_PADRAO, ALPHA_PADRAO

        try:
            apostas = self._gerar_apostas_inteligentes(qtd=qtd, janela=janela, alpha=alpha)
            resposta = self._formatar_resposta(apostas, janela, alpha)
            await update.message.reply_text(resposta, parse_mode='HTML')
        except Exception:
            logger.error("Erro ao gerar apostas:\n" + traceback.format_exc())
            await update.message.reply_text("❌ Erro ao gerar apostas. Tente novamente mais tarde.")

    def _formatar_resposta(self, apostas: List[List[int]], janela: int, alpha: float) -> str:
        """Formata a resposta com as apostas no padrão atual, com rodapé opcional."""
        linhas = ["🎰 <b>SUAS APOSTAS INTELIGENTES</b> 🎰\n"]
        for i, aposta in enumerate(apostas, 1):
            pares = sum(1 for n in aposta if n % 2 == 0)
            linhas.append(
                f"<b>Aposta {i}:</b> {' '.join(f'{n:02d}' for n in aposta)}\n"
                f"🔢 Pares: {pares} | Ímpares: {15 - pares}\n"
            )

        if SHOW_TIMESTAMP:
            try:
                now_sp = datetime.now(ZoneInfo(TIMEZONE))
                carimbo = now_sp.strftime("%Y-%m-%d %H:%M:%S %Z")
            except Exception:
                carimbo = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
            linhas.append(f"<i>janela={janela} | α={alpha:.2f} | {carimbo}</i>")

        return "\n".join(linhas)

    async def meuid(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /meuid – Retorna o ID do usuário"""
        user_id = update.effective_user.id
        await update.message.reply_text(
            f"🆔 Seu ID: <code>{user_id}</code>\n"
            "Use este código para liberação ou autenticação no sistema.",
            parse_mode='HTML'
        )

    async def autorizar(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /autorizar – Adiciona um ID à whitelist (somente admin)"""
        if update.effective_user.id != self.admin_id:
            await update.message.reply_text("⛔ Você não tem permissão para autorizar usuários.")
            return

        if len(context.args) != 1 or not context.args[0].isdigit():
            await update.message.reply_text("⚠️ Uso correto: /autorizar <ID>")
            return

        user_id = int(context.args[0])
        self.whitelist.add(user_id)
        self._salvar_whitelist()
        await update.message.reply_text(f"✅ Usuário {user_id} autorizado com sucesso.")

    async def remover(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /remover – Remove um ID da whitelist (somente admin)"""
        if update.effective_user.id != self.admin_id:
            await update.message.reply_text("⛔ Você não tem permissão para remover usuários.")
            return

        if len(context.args) != 1 or not context.args[0].isdigit():
            await update.message.reply_text("⚠️ Uso correto: /remover <ID>")
            return

        user_id = int(context.args[0])
        if user_id in self.whitelist:
            self.whitelist.remove(user_id)
            self._salvar_whitelist()
            await update.message.reply_text(f"✅ Usuário {user_id} removido da autorização.")
        else:
            await update.message.reply_text(f"ℹ️ Usuário {user_id} não está na whitelist.")

    # ------------- Handler do backtest (revisado) -------------
    async def backtest(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Comando oculto /backtest – somente admin.
        Uso:
          /backtest
          /backtest 200
          /backtest 200 5
          /backtest 200 5 0.35
          /backtest janela=200 bilhetes=5 alpha=0,45
        Limites:
          janela:   50..1000
          bilhetes: 1..20
          alpha:    0.05..0.95
        """
        user_id = update.effective_user.id
        if not self._is_admin(user_id):
            # Silencioso para não expor o comando
            return

        # Parse e validação
        janela, bilhetes_por_concurso, alpha = self._parse_backtest_args(context.args)

        # Confirmação ao admin
        await update.message.reply_text(
            f"Executando backtest com janela={janela}, bilhetes={bilhetes_por_concurso}, alpha={alpha:.2f}..."
        )

        loop = asyncio.get_running_loop()
        try:
            historico = carregar_historico("data/history.csv")

            func = partial(
                executar_backtest_resumido,
                historico=historico,
                janela=janela,
                bilhetes_por_concurso=bilhetes_por_concurso,
                alpha=alpha
            )
            resumo: str = await loop.run_in_executor(None, func)

            # Resposta com truncamento defensivo
            if len(resumo) > 4000:
                resumo = resumo[:4000] + "\n\n[Saída truncada]"
            await update.message.reply_text("📊 BACKTEST (rolling)\n" + resumo)
        except TypeError as te:
            logger.error("Assinatura de executar_backtest_resumido não compatível:\n" + traceback.format_exc())
            await update.message.reply_text(
                "Erro: a função utils.backtest.executar_backtest_resumido não aceita os parâmetros esperados "
                "(historico, janela, bilhetes_por_concurso, alpha). Ajuste a assinatura para compatibilidade."
            )
        except Exception as e:
            logger.error("Erro no backtest:\n" + traceback.format_exc())
            await update.message.reply_text(f"Erro no backtest: {e}")

    def run(self):
        """Inicia o bot"""
        logger.info("Bot iniciado e aguardando comandos.")
        self.app.run_polling()

# ========================
# Execução
# ========================
if __name__ == '__main__':
    bot = LotoFacilBot()
    bot.run()
