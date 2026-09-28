# EmoGol: emociones en comentarios de fútbol con modelos de lenguaje

Trabajo de grado **"Inteligencia emocional en eventos futbolísticos basada en lenguaje natural"**
Ingeniería Electrónica y Telecomunicaciones · Universidad del Cauca
**Autor:** Santiago Fernando Ortega Segura · **Director:** Ph.D. Oscar Mauricio Caicedo Rendón

EmoGol clasifica comentarios de fútbol en español en 7 emociones (alegría, tristeza, enojo, miedo,
sorpresa, burla y neutral) con modelos de lenguaje de pesos abiertos ajustados mediante LoRA/QLoRA.
El trabajo compara 18 modelos (10 modelos de lenguaje, 2 de la familia BERT y 6 clásicos) con los
mismos datos y la misma prueba, y presenta un prototipo que resume las emociones de un partido a
partir de los comentarios de sus videos en YouTube.

## Resultados principales (prueba de 10.000 comentarios)

| Modelo | Accuracy | F1 macro |
|---|---|---|
| Gemma-2 9B + LoRA | 83,6 % | 0,788 |
| **Gemma-2 2B + LoRA (EmoGol-2B)** | **82,5 %** | **0,776** |
| Llama 3.1 8B + LoRA | 79,9 % | 0,753 |
| RoBERTuito (mejor de la familia BERT) | 70,3 % | 0,608 |
| CNN (mejor clásico) | 65,4 % | 0,569 |

EmoGol-2B es el modelo recomendado: su desempeño es estadísticamente equivalente al de Gemma-2 9B
con cerca de un tercio de la memoria (2,3 GB frente a 6,2 GB). La tabla completa de los 18 modelos
está en el repositorio de datos (`final/tabla_maestra_comparativa.csv`).

## Dónde está cada cosa

| Qué | Dónde |
|---|---|
| Código (este repositorio) | Notebooks, scripts de recolección y figuras |
| Datos y resultados | Hugging Face: `santiagortegasegura/emotion-futbol-data` (privado, acceso bajo solicitud) |
| Modelos entrenados (adaptadores LoRA) | Hugging Face: `santiagortegasegura/emotion-futbol-checkpoints` (privado, acceso bajo solicitud) |

Los datos no se guardan en este repositorio.

## Estructura

```
notebooks/        Flujo completo, en orden de ejecución (ver notebooks/README.md)
recoleccion/      youtube_collector.py: recolección de comentarios con la API de YouTube
figuras/          Código de las figuras del Capítulo 4 de la monografía
herramientas/     verificar_secretos.py: revisa que no haya claves antes de subir cambios
requirements.txt  Dependencias para correr los scripts localmente
```

## Flujo del proyecto

1. **Recolección** de comentarios de videos de partidos con la API de YouTube.
2. **Limpieza** y eliminación de duplicados (108.742 → 107.563 comentarios).
3. **Etiquetado automático** con un modelo de lenguaje (Qwen2.5-7B-Instruct, instrucciones few-shot y
   3 votos por comentario), calibrado con una muestra etiquetada a mano.
4. **Filtrado por confianza** (≥ 0,67) y **partición** en 72.000 de entrenamiento, 18.000 de validación
   y 10.000 de prueba (100.000 comentarios).
5. **Ajuste fino** de los 10 modelos de lenguaje (QLoRA) y de los 2 modelos BERT (LoRA);
   **entrenamiento** de los 6 clásicos.
6. **Evaluación** de los 18 modelos sobre la misma prueba: calidad (accuracy, precisión, recall, F1,
   intervalos de confianza) y eficiencia (latencia, memoria, tamaño).
7. **Prototipo EmoGol**: extrae los comentarios de un video de YouTube, los clasifica con EmoGol-2B y
   resume la distribución y la evolución de las emociones.

## Cómo reproducir

Los notebooks están pensados para Google Colab (GPU T4 para entrenamiento, evaluación y prototipo).
Cada uno lee sus entradas de Hugging Face y guarda allí sus resultados. En los secretos de Colab
(ícono de la llave) deben estar:

- `HF_TOKEN`: token de Hugging Face con acceso a los repositorios del proyecto.
- `YOUTUBE_API_KEY`: clave de la API de YouTube Data v3 (solo para recolección y prototipo).

Las claves nunca se escriben en el código. Antes de subir cambios, corre
`python herramientas/verificar_secretos.py`.
