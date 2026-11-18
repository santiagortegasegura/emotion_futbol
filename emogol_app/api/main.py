"""Servicio de inferencia de EmoGol.

Carga emogol.joblib una sola vez al arrancar y clasifica los textos que llegan por HTTP.
Los mensajes entran a una cola y se agrupan en lotes de hasta 16, como en la evaluación:
cuando llegan más mensajes de los que la GPU alcanza a procesar, la cola crece y la
latencia sube, que es lo que miden las pruebas de carga.

Componentes opcionales:
  - Caché en Redis: un mensaje idéntico a uno ya clasificado recibe su emoción sin pasar por la GPU.
  - Métricas en formato Prometheus (/metrics): latencia, cola, lotes, caché, GPU, CPU y RAM.

Variables de entorno:
  EMOGOL_JOBLIB      ruta local del archivo (por defecto emogol.joblib)
  EMOGOL_REPO        repositorio de Hugging Face de donde bajarlo si no existe localmente
  HF_TOKEN           token de Hugging Face (Gemma-2 2B requiere aceptar su licencia)
  YOUTUBE_API_KEY    solo para /analizar_video
  REDIS_URL          por ejemplo redis://localhost:6379/0; si no está, el servicio funciona sin caché
  EMOGOL_CACHE_TTL   segundos que se conserva una emoción en caché (por defecto 86400)
  EMOGOL_ESPERA_MS   espera máxima para completar un lote (por defecto 25 ms)
  EMOGOL_COLA_MAX    mensajes máximos en cola antes de rechazar con 503 (por defecto 5000)
  EMOGOL_SIMULADO    1 = sin GPU, con tiempos simulados (solo para probar el código)
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from emogol.config import NOMBRES

ESPERA_MS = float(os.environ.get("EMOGOL_ESPERA_MS", 25))
COLA_MAX = int(os.environ.get("EMOGOL_COLA_MAX", 5000))
CACHE_TTL = int(os.environ.get("EMOGOL_CACHE_TTL", 86400))


# --- Métricas Prometheus (si la biblioteca no está instalada, el servicio funciona igual) -----------

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

    BUCKETS = (0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144)
    M_LATENCIA = Histogram("emogol_latencia_segundos", "Desde que llega un mensaje hasta que tiene emoción",
                           ["origen"], buckets=BUCKETS)
    M_MENSAJES = Counter("emogol_mensajes_total", "Mensajes clasificados", ["origen"])
    M_COLA = Gauge("emogol_cola_mensajes", "Mensajes esperando en la cola")
    M_LOTE = Histogram("emogol_tamano_lote", "Mensajes por lote enviado a la GPU",
                       buckets=(1, 2, 4, 8, 12, 16))
    M_GPU_USO = Gauge("emogol_gpu_uso_porcentaje", "Uso de la GPU")
    M_GPU_MEM = Gauge("emogol_gpu_memoria_mib", "Memoria de la GPU en uso")
    M_CPU = Gauge("emogol_cpu_porcentaje", "Uso de CPU de la máquina")
    M_RAM = Gauge("emogol_ram_mib", "Memoria RAM en uso de la máquina")
    M_RAM_PROCESO = Gauge("emogol_ram_proceso_mib", "Memoria RAM del proceso del servicio")
    PROMETHEUS = True
except ImportError:
    PROMETHEUS = False


class Telemetria:
    """Lee GPU, CPU y RAM cada segundo y las publica como métricas."""

    def __init__(self, periodo: float = 1.0):
        self.periodo = periodo
        self.ultimo: dict = {}
        self._detener = threading.Event()

    def _gpu(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            return (pynvml.nvmlDeviceGetUtilizationRates(h).gpu,
                    pynvml.nvmlDeviceGetMemoryInfo(h).used / 2**20)
        except Exception:
            return None, None

    def _ciclo(self):
        try:
            import psutil
            proceso = psutil.Process()
            psutil.cpu_percent(None)
        except ImportError:
            psutil = None
        while not self._detener.is_set():
            uso, mem = self._gpu()
            datos = {"gpu_uso_pct": uso, "gpu_memoria_mib": mem}
            if psutil:
                datos.update({"cpu_pct": psutil.cpu_percent(None),
                              "ram_mib": psutil.virtual_memory().used / 2**20,
                              "ram_proceso_mib": proceso.memory_info().rss / 2**20})
            self.ultimo = datos
            if PROMETHEUS:
                for g, k in [(M_GPU_USO, "gpu_uso_pct"), (M_GPU_MEM, "gpu_memoria_mib"), (M_CPU, "cpu_pct"),
                             (M_RAM, "ram_mib"), (M_RAM_PROCESO, "ram_proceso_mib")]:
                    if datos.get(k) is not None:
                        g.set(datos[k])
            self._detener.wait(self.periodo)

    def iniciar(self):
        threading.Thread(target=self._ciclo, daemon=True).start()

    def detener(self):
        self._detener.set()


# --- Caché en Redis ------------------------------------------------------------------------------

class Cache:
    """Guarda la emoción de cada texto exacto. Un texto idéntico produce siempre la misma salida,
    porque la decodificación es voraz; por eso la caché no cambia los resultados."""

    def __init__(self, url: str | None, prefijo: str):
        self.cliente = None
        self.prefijo = prefijo
        self.aciertos = self.fallos = 0
        if url:
            try:
                import redis.asyncio as redis
                self.cliente = redis.from_url(url, decode_responses=True)
            except ImportError:
                self.cliente = None

    @property
    def activa(self) -> bool:
        return self.cliente is not None

    def _clave(self, texto: str) -> str:
        return f"{self.prefijo}:{texto.strip()}"

    async def obtener(self, texto: str) -> str | None:
        if not self.cliente:
            return None
        try:
            valor = await self.cliente.get(self._clave(texto))
        except Exception:
            return None
        if valor is None:
            self.fallos += 1
        else:
            self.aciertos += 1
        return valor

    async def guardar(self, texto: str, etiqueta: str):
        if self.cliente and etiqueta != "invalido":
            try:
                await self.cliente.set(self._clave(texto), etiqueta, ex=CACHE_TTL)
            except Exception:
                pass

    async def vaciar(self):
        if self.cliente:
            async for clave in self.cliente.scan_iter(f"{self.prefijo}:*"):
                await self.cliente.delete(clave)
        self.aciertos = self.fallos = 0


# --- Cola con lotes ------------------------------------------------------------------------------

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
        if PROMETHEUS:
            M_COLA.set(self.cola.qsize())
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
            if PROMETHEUS:
                M_COLA.set(self.cola.qsize())
                M_LOTE.observe(len(lote))
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


async def clasificar_uno(texto: str, usar_cache: bool = True) -> tuple[str, str]:
    """Devuelve (etiqueta, origen), donde origen es 'cache' o 'modelo'."""
    t0 = time.perf_counter()
    cache: Cache = estado["cache"]
    etiqueta, origen = None, "modelo"
    if usar_cache:
        etiqueta = await cache.obtener(texto)
        if etiqueta is not None:
            origen = "cache"
        elif texto in estado["en_curso"]:
            # El mismo texto ya está en la cola: se espera ese resultado en lugar de clasificarlo otra vez.
            etiqueta, origen = await asyncio.shield(estado["en_curso"][texto]), "cache"
            cache.fallos -= 1
            cache.aciertos += 1
    if etiqueta is None:
        if usar_cache:
            futuro = asyncio.get_running_loop().create_future()
            estado["en_curso"][texto] = futuro
            try:
                etiqueta = await estado["cola"].clasificar(texto)
                futuro.set_result(etiqueta)
            except Exception as e:
                futuro.set_exception(e)
                futuro.exception()  # evita el aviso de excepción no recuperada
                raise
            finally:
                estado["en_curso"].pop(texto, None)
            await cache.guardar(texto, etiqueta)
        else:
            etiqueta = await estado["cola"].clasificar(texto)
    if PROMETHEUS:
        M_LATENCIA.labels(origen).observe(time.perf_counter() - t0)
        M_MENSAJES.labels(origen).inc()
    return etiqueta, origen


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    t0 = time.time()
    clasificador = cargar_clasificador()
    huella = clasificador.info().get("huella_adaptador", "simulado")
    estado["clasificador"] = clasificador
    estado["cola"] = ColaInferencia(clasificador, clasificador.config.tamano_lote)
    estado["cola"].iniciar()
    estado["cache"] = Cache(os.environ.get("REDIS_URL"), f"emogol:{huella}")
    estado["en_curso"] = {}
    estado["telemetria"] = Telemetria()
    estado["telemetria"].iniciar()
    estado["segundos_carga"] = round(time.time() - t0, 1)
    yield
    estado["telemetria"].detener()
    estado["cola"].tarea.cancel()


app = FastAPI(title="EmoGol", version="1.1.0", lifespan=ciclo_de_vida)


class Peticion(BaseModel):
    textos: list[str] = Field(..., min_length=1, max_length=1000)
    fuente: str = Field("chat", pattern="^(chat|video)$")
    usar_cache: bool = True


class PeticionVideo(BaseModel):
    video_id: str
    maximo: int = Field(3000, ge=10, le=10000)


@app.get("/salud")
def salud():
    c = estado["clasificador"]
    return {"estado": "ok", "segundos_carga": estado["segundos_carga"], "modelo": c.info(),
            "cache_redis": estado["cache"].activa, "metricas_prometheus": PROMETHEUS}


@app.post("/clasificar")
async def clasificar(p: Peticion):
    c = estado["clasificador"]
    t0 = time.perf_counter()
    validos = c.filtrar(p.textos, p.fuente)
    salidas = await asyncio.gather(*(clasificar_uno(t, p.usar_cache) for t in validos))
    resultado = dict(zip(validos, salidas))
    return {
        "resultados": [{"texto": t,
                        "etiqueta": resultado[t][0] if t in resultado else "descartado",
                        "emocion": NOMBRES.get(resultado[t][0]) if t in resultado else "Descartado (filtro de validez)",
                        "origen": resultado[t][1] if t in resultado else None}
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
    salidas = await asyncio.gather(*(clasificar_uno(d["text"]) for d in datos))
    segundos = time.perf_counter() - t0
    df = pd.DataFrame(datos)
    df["emocion"] = [e for e, _ in salidas]
    r = agregacion.resumen(df, info, segundos)
    evol = agregacion.evolucion(df, info["publicado"], ventana="3h", minimo=50)
    r["evolucion"] = {str(k): v for k, v in evol.to_dict(orient="index").items()}
    r["comentarios"] = df[["created_at", "text", "emocion"]].to_dict(orient="records")
    return r


@app.get("/metricas")
def metricas():
    m = dict(estado["cola"].metricas)
    transcurrido = time.time() - m.pop("inicio")
    cache: Cache = estado["cache"]
    consultas = cache.aciertos + cache.fallos
    m.update({"cola_actual": estado["cola"].cola.qsize(), "segundos": round(transcurrido, 1),
              "mensajes_por_segundo": round(m["procesados"] / transcurrido, 2) if transcurrido else 0,
              "tamano_lote_promedio": round(m["procesados"] / m["lotes"], 2) if m["lotes"] else 0,
              "cache_aciertos": cache.aciertos, "cache_fallos": cache.fallos,
              "cache_tasa_aciertos": round(cache.aciertos / consultas, 4) if consultas else None,
              **estado["telemetria"].ultimo})
    return m


@app.post("/metricas/reiniciar")
def reiniciar():
    estado["cola"].reiniciar_metricas()
    estado["cache"].aciertos = estado["cache"].fallos = 0
    return {"estado": "reiniciadas"}


@app.post("/cache/vaciar")
async def vaciar_cache():
    await estado["cache"].vaciar()
    return {"estado": "vaciada", "activa": estado["cache"].activa}


@app.get("/metrics")
def metrics_prometheus():
    if not PROMETHEUS:
        raise HTTPException(404, "prometheus_client no está instalado.")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
