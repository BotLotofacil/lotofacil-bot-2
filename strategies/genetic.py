# strategies/genetic.py

import random
from typing import List

class GeneticGenerator:
    def __init__(self, historico: List[List[int]]):
        self.historico = historico
        self.populacao_base = historico[-100:] if len(historico) > 100 else historico

    def generate(self, n_apostas: int = 5) -> List[List[int]]:
        """Gera apostas por evolução genética"""
        return [self._gerar_individuo() for _ in range(n_apostas)]

    def _gerar_individuo(self) -> List[int]:
        """Gera uma aposta via crossover + mutação"""
        pai1 = random.choice(self.populacao_base)
        pai2 = random.choice(self.populacao_base)

        # Crossover simples
        corte = random.randint(5, 10)
        filho = list(set(pai1[:corte] + pai2[corte:]))

        # Mutação
        while len(filho) < 15:
            n = random.randint(1, 25)
            if n not in filho:
                filho.append(n)

        # Eliminar duplicatas e garantir 15 números
        filho = list(set(filho))
        return sorted(random.sample(filho, 15))
