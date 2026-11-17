# EmoGol: aplicación desplegable

Esta carpeta convierte EmoGol-2B, que hasta ahora se ejecutaba celda por celda en los notebooks 16 y 17,
en una aplicación con tres piezas que se despliegan por separado:

| Pieza | Archivo | Qué hace |
|---|---|---|
| Archivo ejecutable del modelo | `emogol.joblib` | Configuración, reglas y adaptador LoRA de EmoGol-2B en un solo archivo |
| Servicio de inferencia | `api/main.py` | Carga `emogol.joblib` una vez y expone la API REST |
| Tablero | `tablero/app.py` | Distribución, evolución y ejemplos de un video, a partir de la API |
| Pruebas de carga | `pruebas_carga/generador.py` | Tasas crecientes de mensajes hasta la saturación |

## Estructura

```
emogol_app/
  emogol/                  paquete de Python
    config.py              valores de uso del modelo (los mismos de la evaluación)
    preprocesamiento.py    filtro de validez y antispam (notebooks 16 y 17)
    clasificador.py        EmoGolClasificador, el objeto que se guarda en emogol.joblib
    ingesta.py             comentarios de un video y chat en vivo (YouTube Data API v3)
    agregacion.py          distribución, evolución en el tiempo y ejemplos
  exportar_joblib.py       construye emogol.joblib (notebook 19)
  api/main.py              servicio de inferencia con cola y lotes de 16
  tablero/app.py           tablero en Streamlit
  pruebas_carga/           generador de carga de lazo abierto (notebook 20)
  Dockerfile               imagen del servicio, igual en cualquier proveedor de nube
  requirements.txt, pyproject.toml, .env.example
```

## Uso

```bash
pip install -e .[modelo,ingesta,servicio,tablero]

# 1. Construir el archivo (o descargarlo de huggingface.co/santiagortegasegura/emogol-2b)
python exportar_joblib.py --salida emogol.joblib

# 2. Servicio de inferencia (requiere GPU NVIDIA, HF_TOKEN y, para videos, YOUTUBE_API_KEY)
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 3. Tablero
EMOGOL_API=http://localhost:8000 streamlit run tablero/app.py

# 4. Pruebas de carga
python pruebas_carga/generador.py --url http://localhost:8000 --mensajes chat.csv --duracion 120
```

Desde Python, sin servicio:

```python
import joblib
emogol = joblib.load("emogol.joblib")
emogol.predecir_nombres(["LO BOTO JAJAJAJA", "qué robo del árbitro"])
```

El modelo base Gemma-2 2B no va dentro del archivo: se descarga de Hugging Face al cargarlo, por lo que
hay que aceptar su licencia en `google/gemma-2-2b-it`. En los datos y en el prompt, la emoción desprecio
conserva la etiqueta `burla`; la aplicación muestra "Desprecio".

## API

| Método | Ruta | Entrada | Salida |
|---|---|---|---|
| GET | `/salud` | | Estado y datos del modelo cargado |
| POST | `/clasificar` | `{"textos": [...], "fuente": "chat" o "video"}` | Emoción de cada texto |
| POST | `/analizar_video` | `{"video_id": "...", "maximo": 3000}` | Resumen, evolución y comentarios clasificados |
| GET | `/metricas` | | Mensajes procesados, cola, tamaño de lote, rendimiento |

## Despliegue en la nube

La imagen de `Dockerfile` corre en cualquier máquina con una GPU NVIDIA (por ejemplo, una T4) y el
NVIDIA Container Toolkit. Los secretos se entregan como variables de entorno y nunca se copian a la imagen.

```bash
docker build -t emogol .
docker run --gpus all -p 8000:8000 -e HF_TOKEN=... -e YOUTUBE_API_KEY=... emogol
```

Para probar el código sin GPU, `EMOGOL_SIMULADO=1` usa tiempos simulados. Esos tiempos no son mediciones.
