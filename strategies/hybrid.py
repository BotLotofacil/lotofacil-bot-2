# strategies/hybrid.py

from typing import List
from strategies.genetic import GeneticGenerator
from strategies.stats import StatsGenerator
from strategies.neural import NeuralGenerator

class HybridGenerator:
    def __init__(self, historico: List[List[int]]):
        self.historico = historico
        self.stats_gen = StatsGenerator(historico)
        self.genetic_gen = GeneticGenerator(historico)
        self.neural_gen = NeuralGenerator(historico)

    def generate(self, n_apostas: int = 5) -> List[List[int]]:
        """Gera apostas combinando múltiplas estratégias"""
        apostas = []
        for _ in range(n_apostas):
            aposta = self._combinar_aposta()
            apostas.append(sorted(aposta))
        return apostas

    def _combinar_aposta(self) -> List[int]:
        """Combina uma aposta de 3 estratégias diferentes"""
        stat = self.stats_gen._gerar_aposta()
        gene = self.genetic_gen._gerar_individuo()
        neur = self.neural_gen._simular_aposta()

        # Junta todas as dezenas geradas
        combinado = list(set(stat + gene + neur))

        # Seleciona 15 dezenas mais frequentes entre os 3 conjuntos
        if len(combinado) < 15:
            combinado += [n for n in range(1, 26) if n not in combinado]

        return sorted(combinado[:15])
