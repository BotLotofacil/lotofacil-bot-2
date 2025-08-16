# core/generator.py

import random
from strategies.genetic import GeneticStrategy
from utils.stats import Estatisticas

class ApostaGenerator:
    def __init__(self, historico_path: str = "history_convertido.csv"):
        self.historico_path = historico_path
        self.stats = Estatisticas(historico_path)
        self.strategy = GeneticStrategy(historico_path)

    def gerar_aposta(self) -> list[int]:
        # Gera uma aposta com base em estratégia genética
        aposta = self.strategy.executar()
        if not aposta or len(aposta) != 15:
            # Se falhar, fallback para aleatória
            aposta = self._gerar_aposta_randomica()
        return sorted(aposta)

    def _gerar_aposta_randomica(self) -> list[int]:
        return random.sample(range(1, 26), 15)
