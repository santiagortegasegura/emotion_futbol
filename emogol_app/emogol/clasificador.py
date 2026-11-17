"""EmoGolClasificador: el objeto que se guarda en emogol.joblib.

El archivo .joblib contiene todo lo propio de EmoGol-2B:
  - la configuración de uso (plantilla, recorte a 112 tokens, decodificación voraz, lotes de 16),
  - las reglas de entrada y de lectura de la respuesta,
  - el adaptador LoRA completo (los archivos de `gemma2_2b/adapter_final`, unos 112 MiB).

Lo único que no va dentro es el modelo base Gemma-2 2B de Google, que se descarga de Hugging Face
la primera vez que se carga el clasificador (requiere haber aceptado su licencia y un HF_TOKEN).
Guardar el modelo cuantizado completo con joblib daría un archivo de unos 2 GB que deja de funcionar
cuando cambia la versión de bitsandbytes o de PyTorch; el adaptador, en cambio, es estable.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import time
from pathlib import Path

from .config import ConfigEmoGol
from .preprocesamiento import es_valido


class EmoGolClasificador:
    VERSION = "1.0.0"

    def __init__(self, config: ConfigEmoGol, archivos_adaptador: dict[str, bytes], metadatos: dict | None = None):
        if "adapter_config.json" not in archivos_adaptador:
            raise ValueError("Falta adapter_config.json entre los archivos del adaptador.")
        self.config = config
        self.archivos_adaptador = archivos_adaptador
        self.metadatos = metadatos or {}
        self._modelo = None
        self._tokenizer = None
        self._tokens_comentario = None
        self._dir_adaptador = None

    # --- Serialización: el modelo cargado en GPU nunca se guarda en el .joblib ---------------------

    def __getstate__(self):
        estado = self.__dict__.copy()
        for clave in ("_modelo", "_tokenizer", "_tokens_comentario", "_dir_adaptador", "_bits"):
            estado[clave] = None
        return estado

    def __setstate__(self, estado):
        self.__dict__.update(estado)

    def guardar(self, ruta: str | Path, compresion: int = 3) -> Path:
        import joblib

        ruta = Path(ruta)
        joblib.dump(self, ruta, compress=compresion)
        return ruta

    # --- Carga del modelo ----------------------------------------------------------------------------

    @property
    def cargado(self) -> bool:
        return self._modelo is not None

    def cargar(self, token_hf: str | None = None, bits: int = 4):
        """Carga Gemma-2 2B y le aplica el adaptador guardado en el .joblib.

        bits=4 es la configuración evaluada (NF4, como en el entrenamiento con QLoRA).
        bits=8 (LLM.int8) y bits=16 (float16) solo se usan para comparar el efecto de la cuantización.
        """
        if self.cargado:
            return self
        if bits not in (4, 8, 16):
            raise ValueError("bits debe ser 4, 8 o 16")
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        token_hf = token_hf or os.environ.get("HF_TOKEN")
        self._dir_adaptador = tempfile.mkdtemp(prefix="emogol_adaptador_")
        for nombre, contenido in self.archivos_adaptador.items():
            destino = Path(self._dir_adaptador) / nombre
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_bytes(contenido)

        if bits == 4:
            cuantizacion = {"quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True,
            )}
        elif bits == 8:
            cuantizacion = {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
        else:
            import transformers
            clave = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
            cuantizacion = {clave: torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
        tokenizer = AutoTokenizer.from_pretrained(self.config.modelo_base, token=token_hf)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        base = AutoModelForCausalLM.from_pretrained(
            self.config.modelo_base, **cuantizacion, device_map="auto",
            attn_implementation="eager", token=token_hf)
        modelo = PeftModel.from_pretrained(base, self._dir_adaptador)
        modelo.eval()

        self._tokenizer = tokenizer
        self._modelo = modelo
        self._bits = bits
        self._tokens_comentario = self.config.max_length - len(tokenizer(self.prompt("")).input_ids) - 2
        return self

    # --- Las mismas funciones de los notebooks 10, 16 y 17 --------------------------------------------

    def prompt(self, texto: str) -> str:
        return ("Clasifica la emoción del siguiente comentario de fútbol en una de estas categorías: "
                f"{', '.join(self.config.emociones)}.\n\nComentario: {texto}\n\nEmoción:")

    def recortar(self, texto: str) -> str:
        ids = self._tokenizer(texto, add_special_tokens=False).input_ids
        if len(ids) <= self._tokens_comentario:
            return texto
        return self._tokenizer.decode(ids[:self._tokens_comentario])

    def parsear(self, generado: str) -> str:
        """Regla estricta: la primera palabra debe ser una de las siete emociones."""
        limpio = generado.strip().lower().split()[0] if generado.strip() else ""
        limpio = limpio.strip(".,;:!¡¿?\"'")
        return limpio if limpio in self.config.emociones else "invalido"

    def _predecir_lote(self, textos: list[str]) -> list[str]:
        import torch

        lote = [self.prompt(self.recortar(t)) for t in textos]
        entradas = self._tokenizer(lote, return_tensors="pt", padding=True, truncation=True,
                                   max_length=self.config.max_length).to(self._modelo.device)
        with torch.no_grad():
            salida = self._modelo.generate(**entradas, max_new_tokens=self.config.max_new_tokens,
                                           do_sample=False, pad_token_id=self._tokenizer.pad_token_id)
        largo = entradas["input_ids"].shape[1]
        return [self.parsear(self._tokenizer.decode(salida[j][largo:], skip_special_tokens=True))
                for j in range(len(lote))]

    # --- Interfaz pública ----------------------------------------------------------------------------

    def predecir(self, textos: list[str] | str) -> list[str]:
        """Devuelve la etiqueta de cada texto: alegria, tristeza, enojo, miedo, sorpresa, burla,
        neutral o invalido. La etiqueta `burla` corresponde a la emoción "desprecio"."""
        if isinstance(textos, str):
            textos = [textos]
        if not self.cargado:
            self.cargar()
        resultado = []
        n = self.config.tamano_lote
        for i in range(0, len(textos), n):
            resultado.extend(self._predecir_lote([str(t) for t in textos[i:i + n]]))
        return resultado

    def predecir_nombres(self, textos: list[str] | str) -> list[str]:
        """Igual que predecir, pero con el nombre que ve el usuario (por ejemplo, Desprecio)."""
        return [self.config.nombres.get(e, e) for e in self.predecir(textos)]

    def filtrar(self, textos: list[str], fuente: str = "video") -> list[str]:
        minimo = self.config.min_caracteres_chat if fuente == "chat" else self.config.min_caracteres_video
        return [t for t in textos if es_valido(t, minimo)]

    def info(self) -> dict:
        tam = sum(len(v) for v in self.archivos_adaptador.values())
        return {
            "version": self.VERSION,
            "modelo_base": self.config.modelo_base,
            "adaptador": f"{self.config.repo_adaptador}/{self.config.ruta_adaptador}",
            "tamano_adaptador_mib": round(tam / 2**20, 1),
            "huella_adaptador": huella(self.archivos_adaptador),
            "cargado": self.cargado,
            "bits": getattr(self, "_bits", None),
            **self.metadatos,
        }

    def calentar(self):
        """Primera pasada para que la latencia medida no incluya la compilación inicial."""
        t0 = time.time()
        self.predecir(["Qué golazo, increíble partido"] * self.config.tamano_lote)
        return time.time() - t0


def huella(archivos: dict[str, bytes]) -> str:
    h = hashlib.sha256()
    for nombre in sorted(archivos):
        h.update(nombre.encode())
        h.update(archivos[nombre])
    return h.hexdigest()[:16]
