# strategies/genetic.py

import random
from typing import List

class GeneticStrategy:
    def __init__(self, historico: List[List[int]]):
        self.historico = historico
        self.populacao_base = historico[-100:] if len(historico) > 100 else historico

    def generate(self, n_apostas: int = 5) -> List[List[int]]:
        """Gera múltiplas apostas"""
        return [self._gerar_individuo() for _ in range(n_apostas)]

    def executar(self) -> List[int]:
        """Executa uma única aposta (método padrão)"""
        return self._gerar_individuo()

    def _gerar_individuo(self) -> List[int]:
        pai1 = random.choice(self.populacao_base)
        pai2 = random.choice(self.populacao_base)

        corte = random.randint(5, 10)
        filho = list(set(pai1[:corte] + pai2[corte:]))

        while len(filho) < 15:
            n = random.randint(1, 25)
            if n not in filho:
                filho.append(n)

        filho = list(set(filho))
        return sorted(random.sample(filho, 15))
