# utils/stats.py
from collections import Counter
from typing import List, Dict
from pathlib import Path

# Use o parser robusto já existente no projeto
from .history import carregar_historico

class Estatisticas:
    def __init__(self, historico_path: str):
        """
        Inicializa a classe carregando o histórico de jogos (15 dezenas 1..25).
        Mantém compatibilidade com core/generator.py.
        """
        self.historico_path = historico_path
        self.jogos: List[List[int]] = self._carregar_jogos()

    def _carregar_jogos(self) -> List[List[int]]:
        """
        Lê o histórico com o parser unificado (utils.history.carregar_historico),
        garantindo listas ORDENADAS de 15 números únicos (1..25) por jogo.
        """
        path = Path(self.historico_path)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo de histórico não encontrado: {path}")

        try:
            # carregar_historico retorna List[Set[int]] já validado
            rows = carregar_historico(path)
        except Exception as e:
            raise ValueError(f"Erro ao carregar/interpretar o histórico: {e}")

        jogos: List[List[int]] = []
        for s in rows:
            # s é um set[int] com 15 dezenas válidas
            linha = sorted(int(n) for n in s if 1 <= int(n) <= 25)
            if len(linha) == 15 and len(set(linha)) == 15:
                jogos.append(linha)

        if not jogos:
            raise ValueError("Histórico válido não encontrado (0 linhas com 15 dezenas).")

        return jogos

    def frequencia_numeros(self) -> Dict[int, int]:
        """
        Calcula a frequência de cada número nos jogos carregados.
        Garante retorno com as chaves 1..25, mesmo que frequência=0.
        """
        contagem = Counter()
        for jogo in self.jogos:
            contagem.update(jogo)

        # Garante 1..25 presentes
        freq = {n: contagem.get(n, 0) for n in range(1, 26)}
        return freq

    def numeros_mais_frequentes(self, top_n: int = 15) -> List[int]:
        """
        Retorna os números mais frequentes (empates resolvidos por menor dezena).
        """
        freq = self.frequencia_numeros()
        ordenado = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        return [n for n, _ in ordenado[:max(0, int(top_n))]]

    def numeros_menos_frequentes(self, bottom_n: int = 15) -> List[int]:
        """
        Retorna os números menos frequentes (empates resolvidos por menor dezena).
        """
        freq = self.frequencia_numeros()
        ordenado = sorted(freq.items(), key=lambda x: (x[1], x[0]))
        return [n for n, _ in ordenado[:max(0, int(bottom_n))]]
