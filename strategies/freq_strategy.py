# strategies/freq_strategy.py
import random
from typing import List
from collections import Counter

class FreqStrategy:
    def __init__(self, historico: List[List[int]]):
        self.historico = historico
        self.frequencias = self._calcular_frequencias()

    def generate(self, n_apostas: int = 5) -> List[List[int]]:
        return [self._gerar_aposta() for _ in range(n_apostas)]

    def executar(self) -> List[int]:
        return self._gerar_aposta()

    # --- Internos ---
    def _calcular_frequencias(self) -> Counter:
        numeros = [n for linha in self.historico for n in linha]
        return Counter(numeros)

    def _gerar_aposta(self) -> List[int]:
        # Se não há histórico, cai para aleatório seguro
        if not self.frequencias:
            return sorted(random.sample(range(1, 26), 15))

        # Pega até 20 mais frequentes, mas pode ser <20
        base = [n for n, _ in self.frequencias.most_common(20)]

        # Garante que temos ao menos 15 candidatos
        if len(base) < 15:
            faltam = [n for n in range(1, 26) if n not in base]
            # completa mantendo diversidade
            base.extend(faltam)

        # Agora base tem >=15. Amostra 15 únicos e ordena
        aposta = sorted(random.sample(base, 15))
        return aposta
