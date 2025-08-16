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
        """Executa uma única aposta"""
        return self._gerar_individuo()

    def _gerar_individuo(self) -> List[int]:
        # Seleção dos pais
        pai1 = random.choice(self.populacao_base)
        pai2 = random.choice(self.populacao_base)

        # Crossover
        corte = random.randint(5, 10)
        filho = list(set(pai1[:corte] + pai2[corte:]))

        # Mutação (complementar até 15 números únicos)
        while len(filho) < 15:
            n = random.randint(1, 25)
            if n not in filho:
                filho.append(n)

        # Se por alguma falha ainda estiver menor, completa com números restantes
        if len(filho) < 15:
            restantes = [n for n in range(1, 26) if n not in filho]
            filho.extend(random.sample(restantes, 15 - len(filho)))

        # Garantir exatamente 15 números únicos
        filho = list(set(filho))
        if len(filho) > 15:
            filho = random.sample(filho, 15)

        return sorted(filho)

