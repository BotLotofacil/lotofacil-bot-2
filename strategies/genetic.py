# strategies/genetic.py
import random
from typing import List

class GeneticStrategy:
    def __init__(self, historico: List[List[int]]):
        self.historico = historico
        self.populacao_base = historico[-100:] if len(historico) > 100 else historico

    def generate(self, n_apostas: int = 5) -> List[List[int]]:
        return [self._gerar_individuo() for _ in range(n_apostas)]

    def executar(self) -> List[int]:
        return self._gerar_individuo()

    # --- Interno ---
    def _gerar_individuo(self) -> List[int]:
        # Fallback seguro se histórico vazio
        if not self.populacao_base:
            return sorted(random.sample(range(1, 26), 15))

        pai1 = random.choice(self.populacao_base)
        pai2 = random.choice(self.populacao_base)

        # Garante listas básicas válidas
        p1 = list(dict.fromkeys(int(x) for x in pai1 if 1 <= int(x) <= 25))  # únicos, 1..25
        p2 = list(dict.fromkeys(int(x) for x in pai2 if 1 <= int(x) <= 25))

        # Fallback se algum pai estiver degenerado
        if len(p1) < 5 or len(p2) < 5:
            return sorted(random.sample(range(1, 26), 15))

        # Crossover
        corte = random.randint(5, min(10, len(p1)-1, len(p2)-1))
        filho = list(dict.fromkeys(p1[:corte] + p2[corte:]))

        # Mutação: completa até 15
        pool = [n for n in range(1, 26) if n not in filho]
        while len(filho) < 15 and pool:
            n = random.choice(pool)
            pool.remove(n)
            filho.append(n)

        # Ajuste final
        if len(filho) > 15:
            filho = random.sample(filho, 15)

        return sorted(filho)
