# core/generator.py

import os
import random
import pandas as pd
from strategies.genetic import GeneticStrategy
from utils.stats import Estatisticas
import logging

logger = logging.getLogger(__name__)

class ApostaGenerator:
    def __init__(self, historico_path: str = "data/history.csv"):
        self.historico_path = historico_path

        if not os.path.exists(historico_path):
            raise FileNotFoundError(f"❌ Arquivo de histórico não encontrado: {historico_path}")

        try:
            df = pd.read_csv(historico_path, header=None)
            self.jogos = df.values.tolist()
        except Exception as e:
            logger.error(f"❌ Erro ao carregar histórico: {e}")
            raise

        self.stats = Estatisticas(historico_path)
        self.strategy = GeneticStrategy(self.jogos)

    def gerar_aposta(self) -> list[int]:
        aposta = self.strategy.executar()
        if not aposta or len(aposta) != 15:
            logger.warning("⚠️ Aposta inválida gerada. Usando aleatória.")
            aposta = self._gerar_aposta_randomica()
        return sorted(aposta)

    async def gerar_apostas(self, n_apostas: int = 3) -> list[list[int]]:
        return [self.gerar_aposta() for _ in range(n_apostas)]

    def _gerar_aposta_randomica(self) -> list[int]:
        return random.sample(range(1, 26), 15)






