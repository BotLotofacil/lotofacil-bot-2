# strategies/freq_strategy.py
import random
from typing import List
from collections import Counter

class FreqStrategy:
    def __init__(self, historico: List[List[int]]):
        self.historico = historico
        self.frequencias = self._calcular_frequencias()

    def generate(self, n_apostas: int = 5) -> List[List[int]]:
        """Gera várias apostas baseadas em frequência"""
        return [self._gerar_aposta() for _ in range(n_apostas)]

    def executar(self) -> List[int]:
        """Executa uma única aposta baseada em frequência"""
        return self._gerar_aposta()

    # --- Internos ---
    def _calcular_frequencias(self) -> Counter:
        """Conta a frequência de cada número no histórico"""
        numeros = [n for linha in self.historico for n in linha]
        return Counter(numeros)

    def _gerar_aposta(self) -> List[int]:
        """Gera uma aposta escolhendo os números mais frequentes"""
        mais_frequentes = [n for n, _ in self.frequencias.most_common(20)]
        aposta = random.sample(mais_frequentes, 15)
        return sorted(aposta)

