"""Filtros de entrada: los mismos de los notebooks 16 (videos) y 17 (chat en vivo)."""
import re
from datetime import datetime


def es_valido(texto: str, min_caracteres: int = 10) -> bool:
    """Descarta textos muy cortos o compuestos solo por símbolos o emojis."""
    texto = (texto or "").strip()
    if len(texto) < min_caracteres:
        return False
    if re.fullmatch(r"[^\w\s]+", texto):
        return False
    return True


def normalizar(texto: str) -> str:
    return " ".join(texto.lower().split())


class FiltroAntispam:
    """Descarta un mensaje de 15 o más caracteres idéntico a otro de los últimos 10 minutos.

    Los cantos cortos ("vamos España") se conservan, porque son reacción real del chat.
    """

    def __init__(self, min_caracteres: int = 15, minutos: int = 10):
        self.min_caracteres = min_caracteres
        self.segundos = minutos * 60
        self._ultima_vez: dict[str, datetime] = {}

    def es_spam(self, texto: str, momento: datetime) -> bool:
        clave = normalizar(texto)
        if len(clave) < self.min_caracteres:
            return False
        anterior = self._ultima_vez.get(clave)
        self._ultima_vez[clave] = momento
        return anterior is not None and (momento - anterior).total_seconds() < self.segundos
