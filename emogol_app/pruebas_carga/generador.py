"""Pruebas de carga de EmoGol: tasas crecientes de mensajes hasta la saturación.

Generador de carga de lazo abierto: los mensajes se envían a una tasa fija (mensajes/s),
sin esperar la respuesta del anterior, como llegan los mensajes de un chat en vivo.
Si el servicio no alcanza a clasificarlos, la cola crece, la latencia sube y el
rendimiento logrado se queda por debajo de la tasa ofrecida: ese es el punto de saturación.

Uso:
    python pruebas_carga/generador.py --url http://localhost:8000 --mensajes chat.csv \
        --tasas 0.5 1.12 2 3 4 5 6 8 10 --duracion 120 --salida resultados_carga.csv
"""
from __future__ import annotations

import argparse
import asyncio
import shutil
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, asdict

import httpx
import pandas as pd


class MuestreoGPU:
    """Lee el uso y la memoria de la GPU con nvidia-smi cada medio segundo."""

    def __init__(self, periodo: float = 0.5):
        self.periodo, self.muestras = periodo, []
        self._detener = threading.Event()
        self.disponible = shutil.which("nvidia-smi") is not None

    def _ciclo(self):
        while not self._detener.is_set():
            try:
                s = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                uso, mem = s.stdout.strip().splitlines()[0].split(",")
                self.muestras.append((float(uso), float(mem)))
            except Exception:
                pass
            self._detener.wait(self.periodo)

    def __enter__(self):
        if self.disponible:
            self._hilo = threading.Thread(target=self._ciclo, daemon=True)
            self._hilo.start()
        return self

    def __exit__(self, *a):
        self._detener.set()

    def resumen(self):
        if not self.muestras:
            return None, None
        return (round(statistics.mean(u for u, _ in self.muestras), 1),
                round(max(m for _, m in self.muestras), 0))


@dataclass
class ResultadoNivel:
    escenario: str
    tasa_ofrecida: float
    duracion_s: float
    enviados: int
    completados: int
    errores: int
    rendimiento_logrado: float
    latencia_p50_ms: float | None
    latencia_p95_ms: float | None
    latencia_p99_ms: float | None
    latencia_max_ms: float | None
    cola_maxima: int | None
    lote_promedio: float | None
    uso_gpu_pct: float | None
    memoria_gpu_mib: float | None
    cola_creciente: bool
    saturado: bool


def percentil(valores, p):
    if not valores:
        return None
    v = sorted(valores)
    k = min(len(v) - 1, max(0, round(p / 100 * (len(v) - 1))))
    return round(v[k], 1)


async def _enviar(cliente, url, texto, registros):
    t0 = time.perf_counter()
    try:
        r = await cliente.post(f"{url}/clasificar", json={"textos": [texto], "fuente": "chat"})
        ok = r.status_code == 200
    except Exception:
        ok = False
    t1 = time.perf_counter()
    registros.append({"enviado": t0, "respondido": t1, "ok": ok})


async def nivel(url: str, textos: list[str], tasa: float, duracion: float, escenario: str = "chat",
                limite_p95_ms: float = 10_000, espera_final: float = 300) -> ResultadoNivel:
    registros: list[dict] = []
    limites = httpx.Limits(max_connections=2000, max_keepalive_connections=200)
    async with httpx.AsyncClient(timeout=espera_final, limits=limites) as cliente:
        await cliente.post(f"{url}/metricas/reiniciar")
        with MuestreoGPU() as gpu:
            inicio = time.perf_counter()
            fin = inicio + duracion
            tareas, i = [], 0
            while True:
                objetivo = inicio + i / tasa
                if objetivo >= fin:
                    break
                await asyncio.sleep(max(0.0, objetivo - time.perf_counter()))
                tareas.append(asyncio.create_task(
                    _enviar(cliente, url, textos[i % len(textos)], registros)))
                i += 1
            await asyncio.wait(tareas, timeout=espera_final)
            m = (await cliente.get(f"{url}/metricas")).json()
        uso, mem = gpu.resumen()

    ok = sorted((r for r in registros if r["ok"]), key=lambda r: r["enviado"])
    lat = [1000 * (r["respondido"] - r["enviado"]) for r in ok]
    # Rendimiento: mensajes clasificados entre el primer envío y la última respuesta.
    # Sin saturación es igual a la tasa ofrecida; con saturación, a la capacidad del servicio.
    rendimiento = round(len(ok) / (ok[-1]["respondido"] - inicio), 2) if ok else 0.0
    p95 = percentil(lat, 95)
    errores = i - len(ok)
    # La cola crece si la latencia del último tercio duplica la del primero.
    tercio = max(1, len(lat) // 3)
    cola_creciente = len(lat) >= 6 and statistics.median(lat[-tercio:]) > 2 * statistics.median(lat[:tercio])
    saturado = (rendimiento < 0.9 * tasa or cola_creciente or errores > 0
                or (p95 is not None and p95 > limite_p95_ms))
    return ResultadoNivel(escenario, tasa, duracion, i, len(ok), errores, rendimiento,
                          percentil(lat, 50), p95, percentil(lat, 99), percentil(lat, 100),
                          m.get("cola_maxima"), m.get("tamano_lote_promedio"), uso, mem, cola_creciente, saturado)


async def barrido(url, textos, tasas, duracion, escenario="chat", pausa=15, **kw) -> pd.DataFrame:
    filas = []
    for tasa in tasas:
        r = await nivel(url, textos, tasa, duracion, escenario, **kw)
        filas.append(asdict(r))
        print(f"[{escenario}] {tasa:>5} msg/s -> logrado {r.rendimiento_logrado:>5} msg/s | "
              f"p50 {r.latencia_p50_ms} ms | p95 {r.latencia_p95_ms} ms | cola máx {r.cola_maxima} | "
              f"GPU {r.uso_gpu_pct} % | {'SATURADO' if r.saturado else 'ok'}")
        await asyncio.sleep(pausa)  # deja vaciar la cola antes del siguiente nivel
    return pd.DataFrame(filas)


def clasificar_niveles(df: pd.DataFrame, tasa_real: float = 1.12) -> pd.DataFrame:
    """Etiqueta cada nivel como en el anteproyecto: normal, alta demanda o cercano a la saturación."""
    capacidad = df.loc[~df["saturado"], "rendimiento_logrado"].max() if (~df["saturado"]).any() else 0
    def etiqueta(f):
        if f["saturado"]:
            return "saturación"
        if f["tasa_ofrecida"] <= tasa_real * 1.5:
            return "normal"
        if f["tasa_ofrecida"] >= 0.8 * capacidad:
            return "cercano a la saturación"
        return "alta demanda"
    df = df.copy()
    df["condicion"] = df.apply(etiqueta, axis=1)
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--mensajes", required=True, help="CSV con una columna 'text'")
    p.add_argument("--tasas", type=float, nargs="+", default=[0.5, 1.12, 2, 3, 4, 5, 6, 8, 10])
    p.add_argument("--duracion", type=float, default=120)
    p.add_argument("--escenario", default="chat")
    p.add_argument("--salida", default="resultados_carga.csv")
    a = p.parse_args()
    textos = pd.read_csv(a.mensajes, keep_default_na=False)["text"].astype(str).tolist()
    df = clasificar_niveles(asyncio.run(barrido(a.url, textos, a.tasas, a.duracion, a.escenario)))
    df.to_csv(a.salida, index=False)
    print(f"Guardado: {a.salida}")
