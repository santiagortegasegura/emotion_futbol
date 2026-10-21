"""Recolecta comentarios de videos de YouTube con la YouTube Data API v3.

Uso:
    python recoleccion/youtube_collector.py VIDEO_ID [VIDEO_ID ...]
    python recoleccion/youtube_collector.py --lista videos.txt --max 1000 --evento final_mundial

La clave se lee de la variable YOUTUBE_API_KEY (archivo .env en la raíz del repositorio).
Por cada video se guardan hasta --max comentarios de primer nivel con al menos 10 caracteres
y que no sean solo signos. El resultado es un archivo JSON en datasets/ con los campos que
lee el notebook 01. El campo author no pasa a los conjuntos publicados.
"""
import argparse
import json
import os
import re
from datetime import datetime

from dotenv import load_dotenv
from googleapiclient.discovery import build


def es_comentario_valido(texto):
    texto = texto.strip()
    if len(texto) < 10:
        return False
    if re.fullmatch(r"[^\w\s]+", texto):
        return False
    return True


def recolectar_video(youtube, video_id, evento, max_comentarios):
    datos = []
    vistos = set()
    pagina = None

    while len(datos) < max_comentarios:
        respuesta = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=pagina,
            textFormat="plainText",
        ).execute()

        for item in respuesta["items"]:
            comment_id = item["id"]
            if comment_id in vistos:
                continue
            vistos.add(comment_id)

            comentario = item["snippet"]["topLevelComment"]["snippet"]
            texto = comentario["textDisplay"].strip()
            if not es_comentario_valido(texto):
                continue

            datos.append({
                "comment_id": comment_id,
                "source": "youtube",
                "event": evento,
                "video_id": video_id,
                "author": comentario["authorDisplayName"],
                "created_at": comentario["publishedAt"],
                "likes": comentario["likeCount"],
                "reply_count": item["snippet"]["totalReplyCount"],
                "text": texto,
            })
            if len(datos) >= max_comentarios:
                break

        pagina = respuesta.get("nextPageToken")
        if not pagina:
            break

    return datos


def main():
    parser = argparse.ArgumentParser(description="Recolecta comentarios de videos de YouTube.")
    parser.add_argument("videos", nargs="*", help="IDs de los videos")
    parser.add_argument("--lista", help="archivo de texto con un ID de video por línea")
    parser.add_argument("--evento", default="youtube_match_event", help="nombre del evento")
    parser.add_argument("--max", type=int, default=1000, help="máximo de comentarios por video")
    parser.add_argument("--salida", default="datasets", help="carpeta de salida")
    args = parser.parse_args()

    videos = list(args.videos)
    if args.lista:
        with open(args.lista, encoding="utf-8") as f:
            videos += [linea.strip() for linea in f if linea.strip() and not linea.startswith("#")]
    if not videos:
        parser.error("indique al menos un ID de video o un archivo con --lista")

    load_dotenv()
    youtube = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))

    datos = []
    for video_id in videos:
        comentarios = recolectar_video(youtube, video_id, args.evento, args.max)
        print(f"{video_id}: {len(comentarios)} comentarios")
        datos += comentarios

    os.makedirs(args.salida, exist_ok=True)
    marca = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta = os.path.join(args.salida, f"{args.evento}_{marca}.json")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=4)

    print(f"\nTotal: {len(datos)} comentarios de {len(videos)} videos")
    print(f"Guardado en: {ruta}")


if __name__ == "__main__":
    main()
