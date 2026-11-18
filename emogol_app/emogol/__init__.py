"""EmoGol: clasificación de emociones en comentarios de fútbol en español.

Uso básico:

    import joblib
    emogol = joblib.load("emogol.joblib")
    emogol.predecir(["LO BOTO JAJAJAJA", "qué robo del árbitro"])
"""
from .config import ConfigEmoGol
from .clasificador import EmoGolClasificador

__version__ = "1.0.0"
__all__ = ["ConfigEmoGol", "EmoGolClasificador", "__version__"]
