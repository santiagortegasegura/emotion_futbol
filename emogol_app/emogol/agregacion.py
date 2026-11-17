"""Resumen de la reacción de la afición: distribución, evolución en el tiempo y ejemplos."""
from __future__ import annotations

import pandas as pd

from .config import EMOCIONES, NOMBRES


def distribucion(df: pd.DataFrame, columna: str = "emocion") -> dict:
    validos = df[df[columna] != "invalido"]
    if not len(validos):
        return {}
    pct = validos[columna].value_counts(normalize=True).reindex(EMOCIONES, fill_value=0) * 100
    return {NOMBRES[e]: round(float(v), 2) for e, v in pct.items()}


def evolucion(df: pd.DataFrame, inicio, ventana: str = "3h", minimo: int = 20,
              columna: str = "emocion") -> pd.DataFrame:
    """Porcentaje de cada emoción por ventana de tiempo (3 h para videos, 30 s o 2 min para el chat)."""
    d = df[df[columna] != "invalido"].copy()
    if not len(d):
        return pd.DataFrame()
    d["created_at"] = pd.to_datetime(d["created_at"], utc=True)
    paso = pd.Timedelta(ventana)
    d["ventana"] = ((d["created_at"] - pd.Timestamp(inicio)) // paso) * paso
    conteo = d.groupby("ventana").size()
    d = d[d["ventana"].isin(conteo[conteo >= minimo].index)]
    tabla = (d.groupby("ventana")[columna].value_counts(normalize=True).unstack(fill_value=0)
             .reindex(columns=EMOCIONES, fill_value=0) * 100)
    tabla.columns = [NOMBRES[c] for c in tabla.columns]
    tabla["n"] = conteo.reindex(tabla.index)
    return tabla.round(2)


def ejemplos(df: pd.DataFrame, por_emocion: int = 3, columna: str = "emocion") -> dict:
    orden = "likes" if "likes" in df.columns else "created_at"
    salida = {}
    for e in EMOCIONES:
        d = df[df[columna] == e]
        if len(d):
            salida[NOMBRES[e]] = d.nlargest(por_emocion, orden)["text"].str.slice(0, 140).tolist() \
                if orden == "likes" else d.head(por_emocion)["text"].str.slice(0, 140).tolist()
    return salida


def resumen(df: pd.DataFrame, info: dict | None = None, segundos: float | None = None) -> dict:
    r = {"n_clasificados": int(len(df)),
         "pct_invalidas": round(100 * float((df["emocion"] == "invalido").mean()), 2) if len(df) else None,
         "distribucion_pct": distribucion(df), "ejemplos": ejemplos(df)}
    if segundos:
        r["comentarios_por_segundo"] = round(len(df) / segundos, 2)
    if info:
        r["video"] = info
    return r
