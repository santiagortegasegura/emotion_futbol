# Imagen del servicio de inferencia de EmoGol. Requiere una máquina con GPU NVIDIA
# (por ejemplo, una T4) y el NVIDIA Container Toolkit. Funciona igual en cualquier proveedor de nube.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY emogol/ emogol/
COPY api/ api/
COPY tablero/ tablero/
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# emogol.joblib se descarga de Hugging Face al arrancar (EMOGOL_REPO) o se monta en /app/emogol.joblib.
# Los secretos (HF_TOKEN, YOUTUBE_API_KEY) se entregan como variables de entorno al iniciar el contenedor.
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
