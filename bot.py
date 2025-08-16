# bot.py

import os
import logging
import asyncio
import random
from typing import List
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

from core.generator import ApostaGenerator
from dotenv import load_dotenv

# Carrega variáveis de ambiente de um arquivo .env (apenas localmente)
load_dotenv()

# ========================
# Configuração de Logging
# ========================
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ========================
# Bot Principal
# ========================
class LotoFacilBot:
    def __init__(self):
        self.token = self._get_bot_token()
        self.app = ApplicationBuilder().token(self.token).build()
        self.generator = ApostaGenerator("data/history.csv")
        self._setup_handlers()

    def _get_bot_token(self) -> str:
        """Obtém o token do ambiente"""
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            raise EnvironmentError("❌ Variável TELEGRAM_BOT_TOKEN não configurada.")
        return token

    def _setup_handlers(self):
        """Registra os comandos do bot"""
        self.app.add_handler(CommandHandler("gerar", self.gerar_apostas))
        self.app.add_handler(CommandHandler("meuid", self.meuid))

    async def gerar_apostas(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /gerar – Gera 3 apostas inteligentes"""
        user_id = update.effective_user.id
        logger.info(f"Comando /gerar chamado por {user_id}")

        try:
            apostas = await self.generator.gerar_apostas(n_apostas=3)
            resposta = self._formatar_resposta(apostas)
            await update.message.reply_text(resposta, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Erro ao gerar apostas: {e}")
            await update.message.reply_text("❌ Erro ao gerar apostas. Tente novamente mais tarde.")

    def _formatar_resposta(self, apostas: List[List[int]]) -> str:
        """Formata a resposta com as apostas"""
        resposta = ["🎰 <b>SUAS APOSTAS INTELIGENTES</b> 🎰\n"]
        for i, aposta in enumerate(apostas, 1):
            pares = sum(1 for n in aposta if n % 2 == 0)
            resposta.append(
                f"<b>Aposta {i}:</b> {' '.join(f'{n:02d}' for n in aposta)}\n"
                f"🔢 Pares: {pares} | Ímpares: {15 - pares}\n"
            )
        return "\n".join(resposta)

    async def meuid(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /meuid – Retorna o ID do usuário"""
        user_id = update.effective_user.id
        await update.message.reply_text(
            f"🆔 Seu ID: <code>{user_id}</code>\n"
            "Use este código para liberação ou autenticação no sistema.",
            parse_mode='HTML'
        )

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

