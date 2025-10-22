# strategies/__init__.py
from .genetic import GeneticStrategy
from .freq_strategy import FreqStrategy
from .neural import NeuralStrategy
from .hybrid import HybridGenerator  # <-- ADICIONE

__all__ = ["GeneticStrategy", "FreqStrategy", "NeuralStrategy"]
