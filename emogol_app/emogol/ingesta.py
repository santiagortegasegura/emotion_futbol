"""Ingesta desde YouTube Data API v3: comentarios de un video (notebook 16) y chat en vivo (notebook 17)."""
from __future__ import annotations

import threading
import time
from datetime import datetime

from .preprocesamiento import FiltroAntispam, es_valido


def cliente_youtube(api_key: str):
    from googleapiclient.discovery import build
    return build("youtube", "v3", developerKey=api_key)


def info_video(youtube, video_id: str) -> dict:
    r = youtube.videos().list(part="snippet,statistics,liveStreamingDetails", id=video_id).execute()
    if not r["items"]:
        raise ValueError(f"No se encontró el video {video_id}")
    item = r["items"][0]
    s = item["snippet"]
    return {"video_id": video_id, "titulo": s["title"], "canal": s["channelTitle"],
            "publicado": s["publishedAt"],
            "n_comentarios_total": int(item.get("statistics", {}).get("commentCount", 0)),
            "chat_en_vivo": item.get("liveStreamingDetails", {}).get("activeLiveChatId")}


def comentarios_video(youtube, video_id: str, maximo: int = 3000, min_caracteres: int = 10) -> list[dict]:
    """Comentarios principales del video, del más reciente al más antiguo, con el filtro de validez."""
    datos, vistos, token = [], set(), None
    while len(datos) < maximo:
        r = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=100,
                                          pageToken=token, textFormat="plainText", order="time").execute()
        for item in r["items"]:
            if item["id"] in vistos:
                continue
            vistos.add(item["id"])
            c = item["snippet"]["topLevelComment"]["snippet"]
            texto = c["textDisplay"].strip()
            if es_valido(texto, min_caracteres):
                datos.append({"comment_id": item["id"], "created_at": c["publishedAt"],
                              "likes": c.get("likeCount", 0), "text": texto})
        token = r.get("nextPageToken")
        if not token:
            break
    return datos[:maximo]


class CapturaChatEnVivo:
    """Consulta el chat en vivo cada `intervalo` segundos y guarda los mensajes válidos y no repetidos."""

    def __init__(self, youtube, live_chat_id: str, intervalo: int = 10, min_caracteres: int = 5,
                 antispam: FiltroAntispam | None = None):
        self.youtube, self.live_chat_id, self.intervalo = youtube, live_chat_id, intervalo
        self.min_caracteres = min_caracteres
        self.antispam = antispam or FiltroAntispam()
        self.mensajes: list[dict] = []
        self.registro = {"consultas": 0, "recibidos": 0, "repetidos": 0, "error": None}
        self._vistos: set[str] = set()
        self._detener = threading.Event()
        self._candado = threading.Lock()
        self._hilo: threading.Thread | None = None

    def _ciclo(self, duracion_horas: float):
        token, fin = None, time.time() + duracion_horas * 3600
        while not self._detener.is_set() and time.time() < fin:
            try:
                resp = self.youtube.liveChatMessages().list(liveChatId=self.live_chat_id, part="snippet",
                                                            maxResults=2000, pageToken=token).execute()
            except Exception as e:  # la transmisión terminó o se agotó la cuota
                self.registro["error"] = str(e)[:300]
                break
            self.registro["consultas"] += 1
            token = resp.get("nextPageToken")
            nuevos = []
            for item in resp.get("items", []):
                s = item.get("snippet", {})
                texto = (s.get("displayMessage") or "").strip()
                self.registro["recibidos"] += 1
                if item["id"] in self._vistos or s.get("type") != "textMessageEvent" \
                        or not es_valido(texto, self.min_caracteres):
                    continue
                self._vistos.add(item["id"])
                momento = datetime.fromisoformat(s["publishedAt"].replace("Z", "+00:00"))
                if self.antispam.es_spam(texto, momento):
                    self.registro["repetidos"] += 1
                    continue
                nuevos.append({"comment_id": item["id"], "created_at": s["publishedAt"], "text": texto})
            with self._candado:
                self.mensajes.extend(nuevos)
            self._detener.wait(max(self.intervalo, resp.get("pollingIntervalMillis", 5000) / 1000))

    def iniciar(self, duracion_horas: float = 3):
        self._hilo = threading.Thread(target=self._ciclo, args=(duracion_horas,), daemon=True)
        self._hilo.start()

    def detener(self):
        self._detener.set()
        if self._hilo:
            self._hilo.join(timeout=30)

    def pendientes(self, desde: int) -> list[dict]:
        with self._candado:
            return self.mensajes[desde:]
