import random
from strategies.genetic import GeneticStrategy
from utils.stats import Estatisticas


class ApostaGenerator:
    def __init__(self, historico_path: str = "data/history.csv"):
        self.historico_path = historico_path
        self.stats = Estatisticas(historico_path)
        self.strategy = GeneticStrategy(historico_path)

    def gerar_aposta(self) -> list[int]:
        """
        Gera uma única aposta utilizando a estratégia genética.
        Fallback para aleatória caso a estratégia falhe.
        """
        aposta = self.strategy.executar()
        if not aposta or len(aposta) != 15:
            aposta = self._gerar_aposta_randomica()
        return sorted(aposta)

    def gerar_apostas(self, n_apostas: int = 1) -> list[list[int]]:
        """
        Gera múltiplas apostas válidas.

        :param n_apostas: Quantidade de apostas a gerar.
        :return: Lista de apostas.
        """
        apostas = []
        while len(apostas) < n_apostas:
            nova_aposta = self.gerar_aposta()
            if nova_aposta not in apostas:
                apostas.append(nova_aposta)
        return apostas

    def _gerar_aposta_randomica(self) -> list[int]:
        """
        Gera uma aposta aleatória com 15 números únicos entre 1 e 25.
        """
        return random.sample(range(1, 26), 15)


