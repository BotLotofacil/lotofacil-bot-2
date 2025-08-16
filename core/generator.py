# core/generator.py

import random
import pandas as pd
from strategies.genetic import GeneticStrategy
from utils.stats import Estatisticas

class ApostaGenerator:
    def __init__(self, historico_path: str = "data/history.csv"):
        self.historico_path = historico_path

        # Carrega o CSV como lista de listas
        df = pd.read_csv(historico_path, header=None)
        self.jogos = df.values.tolist()

        self.stats = Estatisticas(historico_path)
        self.strategy = GeneticStrategy(self.jogos)

    def gerar_aposta(self) -> list[int]:
        # Gera uma aposta com base em estratégia genética
        aposta = self.strategy.executar()
        if not aposta or len(aposta) != 15:
            aposta = self._gerar_aposta_randomica()
        return sorted(aposta)

    def gerar_apostas(self, n_apostas: int = 3) -> list[list[int]]:
        return [self.gerar_aposta() for _ in range(n_apostas)]

    def _gerar_aposta_randomica(self) -> list[int]:
        return random.sample(range(1, 26), 15)





