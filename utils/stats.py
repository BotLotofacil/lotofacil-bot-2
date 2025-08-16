from collections import Counter
import pandas as pd
import numpy as np

class Estatisticas:
    def __init__(self, historico_path: str):
        """
        Inicializa a classe carregando o histórico de jogos.

        :param historico_path: Caminho para o arquivo CSV com o histórico.
        """
        self.historico_path = historico_path
        self.jogos = self._carregar_jogos()

    def _carregar_jogos(self) -> list[list[int]]:
        try:
            df = pd.read_csv(self.historico_path, header=None)
            jogos = []

            for linha in df.itertuples(index=False):
                # Cada linha tem uma string como: '1,3,5,...'
                jogo_str = str(linha[0])
                numeros = [int(n.strip()) for n in jogo_str.split(',') if n.strip().isdigit()]
                if len(numeros) == 15:
                    jogos.append(numeros)

            return jogos
        except Exception as e:
            raise ValueError(f"Erro ao carregar o histórico: {e}")

    def frequencia_numeros(self) -> dict:
        """
        Calcula a frequência de cada número nos jogos carregados.

        :return: Dicionário com a frequência de cada número.
        """
        todos_numeros = [num for jogo in self.jogos for num in jogo]
        contagem = Counter(todos_numeros)
        return dict(sorted(contagem.items()))

    def numeros_mais_frequentes(self, top_n: int = 15) -> list[int]:
        """
        Retorna os números mais frequentes.

        :param top_n: Quantidade a retornar.
        :return: Lista de inteiros.
        """
        freq = self.frequencia_numeros()
        return [n for n, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]]

    def numeros_menos_frequentes(self, bottom_n: int = 15) -> list[int]:
        """
        Retorna os números menos frequentes.

        :param bottom_n: Quantidade a retornar.
        :return: Lista de inteiros.
        """
        freq = self.frequencia_numeros()
        return [n for n, _ in sorted(freq.items(), key=lambda x: x[1])[:bottom_n]]

