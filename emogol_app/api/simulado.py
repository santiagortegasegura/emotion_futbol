"""Clasificador simulado para probar el servicio y las pruebas de carga sin GPU.

No mide nada real: el tiempo por lote imita el orden de magnitud de una T4
(unos 5 mensajes/s con lotes llenos). Los resultados de la monografía deben
obtenerse siempre con el modelo real (notebook 20).
"""
import random
import time

from emogol.config import ConfigEmoGol
from emogol.preprocesamiento import es_valido


class ClasificadorSimulado:
    BASE_S, POR_TEXTO_S = 0.35, 0.17

    def __init__(self):
        self.config = ConfigEmoGol()
        self._rng = random.Random(42)

    def predecir(self, textos):
        time.sleep(self.BASE_S + self.POR_TEXTO_S * len(textos))
        return [self._rng.choice(self.config.emociones) for _ in textos]

    def filtrar(self, textos, fuente="chat"):
        m = self.config.min_caracteres_chat if fuente == "chat" else self.config.min_caracteres_video
        return [t for t in textos if es_valido(t, m)]

    def info(self):
        return {"version": "simulado", "cargado": True}

    def calentar(self):
        return 0.0
