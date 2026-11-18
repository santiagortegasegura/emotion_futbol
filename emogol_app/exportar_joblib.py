"""Construye emogol.joblib a partir del adaptador publicado en Hugging Face.

Uso (en Colab o en cualquier máquina, no necesita GPU):
    python exportar_joblib.py --salida emogol.joblib

El resultado es un único archivo con la configuración, las reglas y el adaptador LoRA de EmoGol-2B.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download

from emogol import ConfigEmoGol, EmoGolClasificador


def exportar(salida: str = "emogol.joblib", token: str | None = None) -> Path:
    config = ConfigEmoGol()
    ruta = snapshot_download(repo_id=config.repo_adaptador, repo_type="dataset",
                             allow_patterns=[f"{config.ruta_adaptador}/*"], token=token)
    carpeta = Path(ruta) / config.ruta_adaptador
    archivos = {p.relative_to(carpeta).as_posix(): p.read_bytes()
                for p in sorted(carpeta.rglob("*")) if p.is_file()}
    print(f"Adaptador: {len(archivos)} archivos, {sum(map(len, archivos.values())) / 2**20:.1f} MiB")

    metadatos = {"creado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "origen": f"https://huggingface.co/datasets/{config.repo_adaptador}"}
    clasificador = EmoGolClasificador(config, archivos, metadatos)
    destino = clasificador.guardar(salida)

    info = clasificador.info()
    info["tamano_joblib_mib"] = round(destino.stat().st_size / 2**20, 1)
    info["archivos_adaptador"] = sorted(archivos)
    info["config"] = config.como_dict()
    Path(destino).with_suffix(".json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Guardado: {destino} ({info['tamano_joblib_mib']} MiB) | huella {info['huella_adaptador']}")
    return destino


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--salida", default="emogol.joblib")
    a = p.parse_args()
    exportar(a.salida)
