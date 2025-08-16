# bot.py

import os
import logging
import asyncio
from typing import List, Set
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from core.generator import ApostaGenerator
from dotenv import load_dotenv

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
# Bot Principal
# ========================
class LotoFacilBot:
    def __init__(self):
        self.token = self._get_bot_token()
        self.whitelist_path = "whitelist.txt"
        self.whitelist = self._carregar_whitelist()
        self.app = ApplicationBuilder().token(self.token).build()
        self.generator = ApostaGenerator("data/history.csv")
        self._setup_handlers()

    def _get_bot_token(self) -> str:
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            raise EnvironmentError("❌ Variável TELEGRAM_BOT_TOKEN não configurada.")
        return token

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

    def _setup_handlers(self):
        """Registra os comandos do bot"""
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("gerar", self.gerar_apostas))
        self.app.add_handler(CommandHandler("meuid", self.meuid))
        self.app.add_handler(CommandHandler("autorizar", self.autorizar))
        self.app.add_handler(CommandHandler("remover", self.remover))

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
        """Comando /gerar – Gera 3 apostas inteligentes (com verificação de autorização)"""
        user_id = update.effective_user.id
        logger.info(f"Comando /gerar chamado por {user_id}")

        if user_id not in self.whitelist:
            await update.message.reply_text("⛔ Você não está autorizado a gerar apostas.")
            return

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

    async def autorizar(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /autorizar – Adiciona um ID à whitelist"""
        if len(context.args) != 1 or not context.args[0].isdigit():
            await update.message.reply_text("⚠️ Uso: /autorizar <ID>")
            return
        user_id = int(context.args[0])
        self.whitelist.add(user_id)
        self._salvar_whitelist()
        await update.message.reply_text(f"✅ Usuário {user_id} autorizado.")

    async def remover(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /remover – Remove um ID da whitelist"""
        if len(context.args) != 1 or not context.args[0].isdigit():
            await update.message.reply_text("⚠️ Uso: /remover <ID>")
            return
        user_id = int(context.args[0])
        if user_id in self.whitelist:
            self.whitelist.remove(user_id)
            self._salvar_whitelist()
            await update.message.reply_text(f"✅ Usuário {user_id} removido da autorização.")
        else:
            await update.message.reply_text(f"ℹ️ Usuário {user_id} não está na whitelist.")

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

