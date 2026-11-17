"""Configuración de EmoGol-2B.

Son los mismos valores con que se entrenó y evaluó el modelo (notebooks 08, 10 y 16).
Cambiarlos cambia el comportamiento medido en la monografía.
"""
from dataclasses import dataclass, field, asdict

# Etiquetas tal como aparecen en los datos y en el prompt. No se cambian:
# el modelo aprendió a responder con estas palabras.
EMOCIONES = ("alegria", "tristeza", "enojo", "miedo", "sorpresa", "burla", "neutral")

# Nombres que se muestran al usuario. En la monografía la etiqueta `burla`
# corresponde a la emoción psicológica "desprecio" (díada ira + asco de Plutchik).
NOMBRES = {
    "alegria": "Alegría",
    "tristeza": "Tristeza",
    "enojo": "Enojo",
    "miedo": "Miedo",
    "sorpresa": "Sorpresa",
    "burla": "Desprecio",
    "neutral": "Neutral",
    "invalido": "Inválida",
}


@dataclass
class ConfigEmoGol:
    modelo_base: str = "google/gemma-2-2b-it"
    repo_adaptador: str = "santiagortegasegura/emotion-futbol-checkpoints"
    ruta_adaptador: str = "gemma2_2b/adapter_final"
    emociones: tuple = EMOCIONES
    nombres: dict = field(default_factory=lambda: dict(NOMBRES))
    max_length: int = 112          # tokens de la instrucción completa
    max_new_tokens: int = 6        # decodificación voraz, respuesta corta
    tamano_lote: int = 16
    min_caracteres_video: int = 10  # filtro de validez del notebook 16
    min_caracteres_chat: int = 5    # filtro de validez del notebook 17
    antispam_min_caracteres: int = 15
    antispam_minutos: int = 10

    def como_dict(self) -> dict:
        d = asdict(self)
        d["emociones"] = list(self.emociones)
        return d
