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
        return [self._combinar_aposta() for _ in range(n_apostas)]

    # --- Interno ---
    def _combinar_aposta(self) -> List[int]:
        stat = self.stats_gen.executar()
        gene = self.genetic_gen.executar()
        neur = self.neural_gen.executar()

        usados = set()
        combinado: List[int] = []

        def take(source: List[int], k: int):
            for n in source:
                if 1 <= n <= 25 and n not in usados:
                    combinado.append(n)
                    usados.add(n)
                    if len(combinado) % 5 == 0 and len(combinado) >= k:
                        break

        # quota inicial equilibrada ~5/5/5 (se possível)
        take(stat, 5)
        take(gene, 10)
        take(neur, 15)

        # se ainda faltar, completa com universo
        if len(combinado) < 15:
            for n in range(1, 26):
                if n not in usados:
                    combinado.append(n)
                    usados.add(n)
                    if len(combinado) == 15:
                        break

        return sorted(combinado[:15])
