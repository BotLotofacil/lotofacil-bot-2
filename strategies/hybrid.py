# strategies/hybrid.py
from typing import List
from .genetic import GeneticStrategy
from .freq_strategy import FreqStrategy
from .neural import NeuralStrategy

class HybridGenerator:
    def __init__(self, historico: List[List[int]]):
        self.historico = historico
        self.stats_gen = FreqStrategy(historico)
        self.genetic_gen = GeneticStrategy(historico)
        self.neural_gen = NeuralStrategy(historico)

    def generate(self, n_apostas: int = 5) -> List[List[int]]:
        """Gera apostas combinando múltiplas estratégias"""
        return [self._combinar_aposta() for _ in range(n_apostas)]

    # --- Interno ---
    def _combinar_aposta(self) -> List[int]:
        """
        Combina uma aposta de 3 estratégias diferentes.
        Usa as saídas públicas (executar) e consolida 15 dezenas.
        """
        stat = self.stats_gen.executar()
        gene = self.genetic_gen.executar()
        neur = self.neural_gen.executar()

        combinado = list(set(stat + gene + neur))
        if len(combinado) < 15:
            combinado += [n for n in range(1, 26) if n not in combinado]

        return sorted(combinado[:15])
