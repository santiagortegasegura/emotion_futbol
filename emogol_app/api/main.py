"""Servicio de inferencia de EmoGol.

Carga emogol.joblib una sola vez al arrancar y clasifica los textos que llegan por HTTP.
Los mensajes entran a una cola y se agrupan en lotes de hasta 16, como en la evaluación:
así, cuando llegan más mensajes de los que la GPU alcanza a procesar, la cola crece y la
latencia sube, que es lo que miden las pruebas de carga.

Variables de entorno:
  EMOGOL_JOBLIB      ruta local del archivo (por defecto emogol.joblib)
  EMOGOL_REPO        repositorio de Hugging Face de donde bajarlo si no existe localmente
  HF_TOKEN           token de Hugging Face (Gemma-2 2B requiere aceptar su licencia)
  YOUTUBE_API_KEY    solo para /analizar_video
  EMOGOL_ESPERA_MS   espera máxima para completar un lote (por defecto 25 ms)
  EMOGOL_COLA_MAX    mensajes máximos en cola antes de rechazar con 503 (por defecto 5000)
  EMOGOL_SIMULADO    1 = sin GPU, con tiempos simulados (solo para probar el código)
"""
from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from emogol.config import NOMBRES

ESPERA_MS = float(os.environ.get("EMOGOL_ESPERA_MS", 25))
COLA_MAX = int(os.environ.get("EMOGOL_COLA_MAX", 5000))


def cargar_clasificador():
    if os.environ.get("EMOGOL_SIMULADO") == "1":
        from api.simulado import ClasificadorSimulado
        return ClasificadorSimulado()
    ruta = Path(os.environ.get("EMOGOL_JOBLIB", "emogol.joblib"))
    if not ruta.exists():
        from huggingface_hub import hf_hub_download
        repo = os.environ.get("EMOGOL_REPO", "santiagortegasegura/emogol-2b")
        ruta = Path(hf_hub_download(repo_id=repo, filename="emogol.joblib", token=os.environ.get("HF_TOKEN")))
    clasificador = joblib.load(ruta)
    clasificador.cargar()
    clasificador.calentar()
    return clasificador


class ColaInferencia:
    """Agrupa los mensajes que llegan en lotes y los pasa al modelo en un solo hilo (una GPU)."""

    def __init__(self, clasificador, tamano_lote: int):
        self.clasificador = clasificador
        self.tamano_lote = tamano_lote
        self.cola: asyncio.Queue = asyncio.Queue()
        self.ejecutor = ThreadPoolExecutor(max_workers=1)
        self.tarea: asyncio.Task | None = None
        self.reiniciar_metricas()

    def reiniciar_metricas(self):
        self.metricas = {"procesados": 0, "lotes": 0, "segundos_modelo": 0.0, "rechazados": 0,
                         "cola_maxima": 0, "inicio": time.time()}

    async def clasificar(self, texto: str) -> str:
        if self.cola.qsize() >= COLA_MAX:
            self.metricas["rechazados"] += 1
            raise HTTPException(503, "Servicio saturado: la cola de mensajes está llena.")
        futuro = asyncio.get_running_loop().create_future()
        await self.cola.put((texto, futuro))
        self.metricas["cola_maxima"] = max(self.metricas["cola_maxima"], self.cola.qsize())
        return await futuro

    async def _trabajar(self):
        bucle = asyncio.get_running_loop()
        while True:
            lote = [await self.cola.get()]
            limite = bucle.time() + ESPERA_MS / 1000
            while len(lote) < self.tamano_lote:
                restante = limite - bucle.time()
                if restante <= 0:
                    break
                try:
                    lote.append(await asyncio.wait_for(self.cola.get(), restante))
                except asyncio.TimeoutError:
                    break
            textos = [t for t, _ in lote]
            t0 = time.perf_counter()
            try:
                etiquetas = await bucle.run_in_executor(self.ejecutor, self.clasificador.predecir, textos)
            except Exception as e:  # un error del modelo no debe tumbar el servicio
                for _, f in lote:
                    if not f.done():
                        f.set_exception(e)
                continue
            self.metricas["segundos_modelo"] += time.perf_counter() - t0
            self.metricas["procesados"] += len(lote)
            self.metricas["lotes"] += 1
            for (_, f), e in zip(lote, etiquetas):
                if not f.done():
                    f.set_result(e)

    def iniciar(self):
        self.tarea = asyncio.create_task(self._trabajar())


estado: dict = {}


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    t0 = time.time()
    clasificador = cargar_clasificador()
    estado["clasificador"] = clasificador
    estado["cola"] = ColaInferencia(clasificador, clasificador.config.tamano_lote)
    estado["cola"].iniciar()
    estado["segundos_carga"] = round(time.time() - t0, 1)
    yield
    estado["cola"].tarea.cancel()


app = FastAPI(title="EmoGol", version="1.0.0", lifespan=ciclo_de_vida)


class Peticion(BaseModel):
    textos: list[str] = Field(..., min_length=1, max_length=1000)
    fuente: str = Field("chat", pattern="^(chat|video)$")


class PeticionVideo(BaseModel):
    video_id: str
    maximo: int = Field(3000, ge=10, le=10000)


@app.get("/salud")
def salud():
    c = estado["clasificador"]
    return {"estado": "ok", "segundos_carga": estado["segundos_carga"], "modelo": c.info()}


@app.post("/clasificar")
async def clasificar(p: Peticion):
    c = estado["clasificador"]
    t0 = time.perf_counter()
    validos = c.filtrar(p.textos, p.fuente)
    etiquetas = await asyncio.gather(*(estado["cola"].clasificar(t) for t in validos))
    resultado = dict(zip(validos, etiquetas))
    return {
        "resultados": [{"texto": t, "etiqueta": resultado.get(t, "descartado"),
                        "emocion": NOMBRES.get(resultado.get(t), "Descartado (filtro de validez)")}
                       for t in p.textos],
        "milisegundos": round(1000 * (time.perf_counter() - t0), 1),
    }


@app.post("/analizar_video")
async def analizar_video(p: PeticionVideo):
    import pandas as pd
    from emogol import agregacion, ingesta

    clave = os.environ.get("YOUTUBE_API_KEY")
    if not clave:
        raise HTTPException(500, "Falta YOUTUBE_API_KEY en el entorno.")
    yt = ingesta.cliente_youtube(clave)
    info = await asyncio.to_thread(ingesta.info_video, yt, p.video_id)
    datos = await asyncio.to_thread(ingesta.comentarios_video, yt, p.video_id, p.maximo)
    if not datos:
        raise HTTPException(404, "El video no tiene comentarios válidos.")
    t0 = time.perf_counter()
    etiquetas = await asyncio.gather(*(estado["cola"].clasificar(d["text"]) for d in datos))
    segundos = time.perf_counter() - t0
    df = pd.DataFrame(datos)
    df["emocion"] = etiquetas
    r = agregacion.resumen(df, info, segundos)
    evol = agregacion.evolucion(df, info["publicado"], ventana="3h", minimo=50)
    r["evolucion"] = {str(k): v for k, v in evol.to_dict(orient="index").items()}
    r["comentarios"] = df[["created_at", "text", "emocion"]].to_dict(orient="records")
    return r


@app.get("/metricas")
def metricas():
    m = dict(estado["cola"].metricas)
    transcurrido = time.time() - m.pop("inicio")
    m.update({"cola_actual": estado["cola"].cola.qsize(), "segundos": round(transcurrido, 1),
              "mensajes_por_segundo": round(m["procesados"] / transcurrido, 2) if transcurrido else 0,
              "tamano_lote_promedio": round(m["procesados"] / m["lotes"], 2) if m["lotes"] else 0})
    return m


@app.post("/metricas/reiniciar")
def reiniciar():
    estado["cola"].reiniciar_metricas()
    return {"estado": "reiniciadas"}
