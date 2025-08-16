# utils/stats.py

from collections import Counter
import numpy as np
import pandas as pd


class Estatisticas:
    @staticmethod
    def frequencia_numeros(jogos: list[list[int]]) -> dict:
        """
        Calcula a frequência de cada número nos jogos fornecidos.

        :param jogos: Lista de jogos, onde cada jogo é uma lista de inteiros.
        :return: Dicionário com a frequência de cada número.
        """
        todos_numeros = [num for jogo in jogos for num in jogo]
        contagem = Counter(todos_numeros)
        return dict(sorted(contagem.items()))

    @staticmethod
    def numeros_mais_frequentes(jogos: list[list[int]], top_n: int = 15) -> list[int]:
        """
        Retorna os números mais frequentes nos jogos.

        :param jogos: Lista de jogos.
        :param top_n: Quantidade de números mais frequentes a retornar.
        :return: Lista de inteiros.
        """
        frequencia = Estatisticas.frequencia_numeros(jogos)
        return [numero for numero, _ in sorted(frequencia.items(), key=lambda item: item[1], reverse=True)[:top_n]]

    @staticmethod
    def numeros_menos_frequentes(jogos: list[list[int]], bottom_n: int = 15) -> list[int]:
        """
        Retorna os números menos frequentes nos jogos.

        :param jogos: Lista de jogos.
        :param bottom_n: Quantidade de números menos frequentes a retornar.
        :return: Lista de inteiros.
        """
        frequencia = Estatisticas.frequencia_numeros(jogos)
        return [numero for numero, _ in sorted(frequencia.items(), key=lambda item: item[1])[:bottom_n]]
