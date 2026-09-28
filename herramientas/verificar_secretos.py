"""Revisa que ningún archivo del repositorio contenga claves o tokens antes de subirlo a GitHub.

Uso:  python herramientas/verificar_secretos.py
Revisa también las salidas guardadas dentro de los notebooks (.ipynb), donde suelen quedar claves
pegadas por error o mensajes de error que las muestran.
"""
import pathlib
import re
import sys

PATRONES = {
    "clave de Google/YouTube": re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    "token de Hugging Face": re.compile(r"hf_[A-Za-z0-9]{30,}"),
    "token de GitHub": re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    "clave de OpenAI/Anthropic": re.compile(r"sk-[A-Za-z0-9_\-]{30,}"),
}
IGNORAR = {".git", "venv", ".venv", "__pycache__", ".ipynb_checkpoints"}

raiz = pathlib.Path(__file__).resolve().parent.parent
hallazgos = 0
for archivo in raiz.rglob("*"):
    if not archivo.is_file() or IGNORAR & set(archivo.parts) or archivo.stat().st_size > 50_000_000:
        continue
    try:
        texto = archivo.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        continue
    for n, linea in enumerate(texto.splitlines(), 1):
        for nombre, patron in PATRONES.items():
            for m in patron.finditer(linea):
                hallazgos += 1
                print(f"[ALERTA] {archivo.relative_to(raiz)}:{n}  {nombre}: {m.group()[:8]}… (oculta)")

if hallazgos:
    print(f"\n{hallazgos} posible(s) clave(s). Bórralas del archivo (y de las salidas del notebook) antes de subir.")
    sys.exit(1)
print("Sin claves ni tokens. Se puede subir.")
