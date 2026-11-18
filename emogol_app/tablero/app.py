"""Tablero de EmoGol (Streamlit).

Consume la API del servicio de inferencia; no carga el modelo por sí mismo.
    EMOGOL_API=http://localhost:8000 streamlit run tablero/app.py
"""
import os

import httpx
import pandas as pd
import streamlit as st

API = os.environ.get("EMOGOL_API", "http://localhost:8000")
ORDEN = ["Alegría", "Enojo", "Desprecio", "Neutral", "Tristeza", "Sorpresa", "Miedo"]

st.set_page_config(page_title="EmoGol", layout="wide")
st.title("EmoGol: emociones de la afición")

try:
    salud = httpx.get(f"{API}/salud", timeout=5).json()
    st.caption(f"Servicio activo · modelo {salud['modelo'].get('version')} · {API}")
except Exception:
    st.error(f"No hay conexión con el servicio en {API}.")
    st.stop()

pestana_video, pestana_texto, pestana_metricas = st.tabs(["Analizar un video", "Probar textos", "Métricas"])

with pestana_video:
    video = st.text_input("ID o enlace del video de YouTube", placeholder="https://www.youtube.com/watch?v=...")
    maximo = st.slider("Comentarios a analizar", 100, 5000, 3000, step=100)
    if st.button("Analizar", type="primary") and video:
        video_id = video.split("v=")[-1].split("&")[0].split("/")[-1]
        with st.spinner("Descargando y clasificando comentarios..."):
            r = httpx.post(f"{API}/analizar_video", json={"video_id": video_id, "maximo": maximo}, timeout=3600)
        if r.status_code != 200:
            st.error(r.json().get("detail", r.text))
        else:
            d = r.json()
            st.subheader(d["video"]["titulo"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Comentarios clasificados", f"{d['n_clasificados']:,}".replace(",", "."))
            c2.metric("Comentarios por segundo", d.get("comentarios_por_segundo"))
            c3.metric("Respuestas inválidas", f"{d['pct_invalidas']} %")
            dist = pd.Series(d["distribucion_pct"]).reindex(ORDEN).fillna(0)
            st.markdown("**Distribución de emociones (% de comentarios)**")
            st.bar_chart(dist, horizontal=True)
            if d.get("evolucion"):
                st.markdown("**Evolución en ventanas de 3 horas desde la publicación**")
                evol = pd.DataFrame(d["evolucion"]).T.drop(columns="n", errors="ignore")
                st.line_chart(evol[[c for c in ORDEN if c in evol]])
            st.markdown("**Ejemplos por emoción**")
            for emocion, textos in d["ejemplos"].items():
                with st.expander(emocion):
                    for t in textos:
                        st.write(f"· {t}")
            st.download_button("Descargar comentarios clasificados (CSV)",
                               pd.DataFrame(d["comentarios"]).to_csv(index=False), "emogol_comentarios.csv")

with pestana_texto:
    textos = st.text_area("Un comentario por línea", "LO BOTO JAJAJAJA\nqué robo del árbitro\ntapalo tapalo")
    if st.button("Clasificar"):
        lineas = [t for t in textos.splitlines() if t.strip()]
        r = httpx.post(f"{API}/clasificar", json={"textos": lineas, "fuente": "chat"}, timeout=120).json()
        st.dataframe(pd.DataFrame(r["resultados"])[["texto", "emocion"]], use_container_width=True)
        st.caption(f"{r['milisegundos']} ms")

with pestana_metricas:
    st.json(httpx.get(f"{API}/metricas", timeout=5).json())
