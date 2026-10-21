# EmoGol: emociones de la afición en comentarios de fútbol en español

Código del trabajo de grado **"Inteligencia emocional en eventos futbolísticos basada en lenguaje natural"**
(Universidad del Cauca, Facultad de Ingeniería Electrónica y Telecomunicaciones, 2026).

**Autor:** Santiago Fernando Ortega Segura · **Director:** Ph.D. Oscar Mauricio Caicedo Rendón

EmoGol clasifica comentarios de fútbol en español en siete emociones (alegría, tristeza, enojo, miedo,
sorpresa, burla y neutral) y resume la reacción de la afición frente a un partido. Su modelo, **EmoGol-2B**,
es Gemma-2 2B ajustado con QLoRA sobre un conjunto de 100.000 comentarios de YouTube.

## Resultados principales

Sobre una prueba de 10.000 comentarios que no participó en el entrenamiento:

| Modelo | Accuracy | F1 macro [IC 95 %] | Memoria GPU (MiB) |
|---|---|---|---|
| **EmoGol-2B** (Gemma-2 2B + QLoRA, seleccionado) | **0,825** | **0,776** [0,748; 0,799] | 2.273 |
| Gemma-2 9B + QLoRA (el más preciso de los 17) | 0,837 | 0,788 [0,761; 0,810] | 6.212 |
| RoBERTuito + LoRA (mejor modelo BERT) | 0,703 | 0,608 | 946 |
| CNN (mejor modelo clásico) | 0,654 | 0,569 | 87 |
| DeepSeek-V4.1-Flash sin ajuste fino (mejor de sus tres modos) | 0,660 | 0,558 [0,535; 0,580] | API comercial |

- EmoGol-2B y Gemma-2 9B no se distinguen estadísticamente en F1 macro, pero EmoGol-2B necesita 2,7 veces
  menos memoria y cuesta 4,1 veces menos por comentario.
- EmoGol-2B clasifica unos 14.000 comentarios por hora en una GPU Tesla T4, a menos de 4 centavos de dólar
  por cada 1.000 comentarios en la nube.
- El prototipo analizó la final de la Copa del Mundo 2026 (3.000 comentarios en unos 14 minutos) y siguió en
  vivo el chat de la final del Mundial Femenino Sub-20 2026.

El análisis completo está en la monografía.

## Uso rápido

Para clasificar comentarios con EmoGol-2B, la tarjeta del repositorio
[`emotion-futbol-checkpoints`](https://huggingface.co/datasets/santiagortegasegura/emotion-futbol-checkpoints)
tiene un ejemplo de pocas líneas. Para analizar todos los comentarios de un video de YouTube, usa el notebook 16.

## Estructura del repositorio

| Carpeta | Contenido |
|---|---|
| `notebooks/` | Los 18 notebooks del flujo final, numerados en orden de ejecución (tabla de abajo). |
| `recoleccion/` | `youtube_collector.py`: extracción de comentarios de YouTube con la YouTube Data API v3. |
| `figuras/` | Código de los diagramas de la monografía (Figuras 3.1, 3.2, 3.3 y 3.5). |
| `herramientas/` | `verificar_secretos.py`: comprobación de los secretos de Colab antes de ejecutar. |
| `requirements.txt` | Dependencias de Python. |

## Datos y modelos

Los datos y los modelos están en Hugging Face Hub, en la cuenta
[santiagortegasegura](https://huggingface.co/santiagortegasegura), y son de acceso público con fines
académicos:

- **`emotion-futbol-data`**: 72.000 comentarios de entrenamiento, 18.000 de validación, la prueba de 10.000
  comentarios, los resultados de los modelos con la tabla comparativa y las gráficas (`final/`), las
  predicciones y matrices de confusión de los modelos de lenguaje, los resultados de los dos casos de estudio
  (`prototipo/mundial2026_prueba/` y `prototipo/en_vivo_prueba/`) y los de la comparación con DeepSeek
  (`final/comercial/`).
- **`emotion-futbol-checkpoints`**: los adaptadores LoRA de los modelos de lenguaje, entre ellos el de
  EmoGol-2B (`gemma2_2b/adapter_final`), y los modelos ajustados de RoBERTuito y BETO. Cada adaptador se
  combina con el modelo base público correspondiente.

## Notebooks

En orden de ejecución. Todos corren en Google Colab. Los notebooks 03, 07 a 10, 12 a 14, 16 y 17 usan una GPU
Tesla T4; los demás corren en CPU. Los notebooks 01 a 05 leen el conjunto crudo desde Google Drive (ver notas);
desde el 06, todos los datos se leen de Hugging Face.

| Nº | Notebook | Qué hace | Nombre durante el desarrollo |
|---|---|---|---|
| 01 | `01_Preparacion_Datos.ipynb` | Limpieza y deduplicación (108.742 → 107.563) y muestra para etiquetado manual | `01_Preparacion_de_Datos` |
| 02 | `02_Refuerzo_Clase_Miedo.ipynb` | Refuerzo de la clase miedo en la muestra manual; pool de 106.923 | `01b_Enriquecimiento_Miedo` |
| 03 | `03_Etiquetado_Automatico.ipynb` | Calibración del anotador y etiquetado del pool (3 votos, confianza ≥ 0,67) | `02_Anotacion` |
| 04 | `04_Particion_Entrenamiento_Validacion.ipynb` | Partición estratificada: 72.000 entrenamiento y 18.000 validación | `04b_0_Preparar_Split_90k` |
| 05 | `05_Datos_Hugging_Face.ipynb` | Subida de los datos a Hugging Face | `05_migrar_datos_a_huggingface` |
| 06 | `06_Prueba_Final.ipynb` | Conjunto de prueba de 10.000 comentarios | `04c_Prueba_Final` |
| 07 | `07_Ajuste_Fino_Unsloth.ipynb` | Ajuste fino QLoRA de 7 modelos de lenguaje (ejecución: Llama 3.1 8B) | `06_entrenamiento_real_unsloth` |
| 08 | `08_Ajuste_Fino_Gemma.ipynb` | Ajuste fino QLoRA de Gemma-2 2B y 9B (ejecución: Gemma-2 2B, EmoGol-2B) | `06_entrenamiento_real_gemma_sin_unsloth` |
| 09 | `09_Evaluacion_Modelos_Lenguaje.ipynb` | Eficiencia: latencia, memoria y tamaño (ejecución: Mistral 7B) | `07_Evaluacion_Final_Split_Oficial` |
| 10 | `10_Metricas_Prueba_Final.ipynb` | Métricas de calidad oficiales sobre la prueba final (ejecución: Gemma-2 2B) | `07b_Evaluacion_Prueba_Final` |
| 11 | `11_Sensibilidad_Interpretacion.ipynb` | Verificación de la regla de interpretación de respuestas | `07c_Sensibilidad_Parseo` |
| 12 | `12_Modelos_Clasicos.ipynb` | SVM, árbol, Random Forest, CNN, BiLSTM y GNN | `08_Clasicos_Split_Oficial` |
| 13 | `13_BERT_RoBERTuito.ipynb` | RoBERTuito con LoRA | `09_BERT_Family_Split_Oficial` |
| 14 | `14_BERT_BETO.ipynb` | BETO con LoRA (mismo notebook que el 13) | `09_BERT_Family_Split_Oficial` |
| 15 | `15_Tabla_Maestra_Graficas.ipynb` | Tabla maestra de los modelos y gráficas (incluye Qwen2.5 7B; ver notas) | `11_Consolidacion_Graficas` |
| 16 | `16_Prototipo_EmoGol.ipynb` | Prototipo EmoGol sobre un video de YouTube (final de la Copa del Mundo 2026) | `10_Clasificador_Comentarios_Video` |
| 17 | `17_EmoGol_Chat_En_Vivo.ipynb` | Captura y clasificación del chat en vivo de una transmisión (final del Mundial Femenino Sub-20 2026) | `10b_EmoGol_Chat_En_Vivo` |
| 18 | `18_Comparacion_Modelo_Comercial.ipynb` | Comparación con DeepSeek-V4.1-Flash sin ajuste fino (zero-shot, con guía y con ejemplos) sobre la misma prueba | `18_Comparacion_Modelo_Comercial` |

### Secretos de Colab

Las claves se leen desde los secretos de Colab y nunca se escriben en el código:

| Secreto | Notebooks |
|---|---|
| `HF_TOKEN` | Todos |
| `YOUTUBE_API_KEY` | 16 y 17 |
| `DEEPSEEK_API_KEY` | 18 |

Para correr localmente el script de recolección, crea un archivo `.env` en la raíz con `YOUTUBE_API_KEY=...`
(el archivo está excluido en `.gitignore`).

## Notas

- **Datos crudos.** Los notebooks 01 a 05 leen el conjunto crudo desde Google Drive (`RUTA/A` es la carpeta que
  se debe reemplazar por la propia). El conjunto crudo no se publica porque incluye los nombres de usuario de
  YouTube; los archivos publicados en Hugging Face no incluyen nombres de usuario.
- **Reproducción.** Para volver a entrenar o evaluar, cambia `TU_USUARIO_HF` por tu propia cuenta de Hugging Face,
  de modo que los resultados se guarden en tus repositorios.
- **Numeración.** Los notebooks se renumeraron en orden de ejecución para su publicación. Las salidas
  conservan los mensajes originales, así que en algunas aparece el nombre que tenían durante el desarrollo
  (columna de la derecha). Por la misma razón, los archivos de resultados en Hugging Face conservan sus
  prefijos originales: `07_resultados_finales.csv` (notebooks 09 y 10), `08_resultados_clasicos.csv`
  (notebook 12) y `09_resultados_bert_family.csv` (notebooks 13 y 14). Los archivos de datos
  `04b_entrenamiento_72k.csv` y `04b_validacion_interna_18k.csv` los genera el notebook 04.
- **Mensajes.** Para la publicación se ajustó la redacción de algunos mensajes informativos de los notebooks
  (en el código y en las salidas guardadas). El código y los resultados no cambiaron.
- **Varias sesiones.** El etiquetado del pool (03) y los entrenamientos (07 y 08) se pueden retomar desde el
  último punto guardado, así que se hicieron en varias sesiones de Colab. De los entrenamientos se incluye
  la sesión en la que terminó un modelo por notebook.
- **Una ejecución por modelo.** Los notebooks 07, 08, 09 y 10 se corren una vez por modelo cambiando
  `CLAVE_MODELO`; aquí se incluye una ejecución de ejemplo de cada uno.
- **Resultados de los modelos de lenguaje.** La eficiencia (latencia, memoria, tamaño) viene del 09 y las
  métricas de calidad oficiales del 10. El 15 consolida los resultados de todos los modelos entrenados.
- **Qwen2.5 7B.** Se entrenó y evaluó con la misma receta (sus archivos están en Hugging Face con la clave
  `qwen_7b` y aparece en la tabla maestra del notebook 15), pero la monografía lo excluye de la comparación
  final de 17 modelos, porque Qwen2.5-7B-Instruct es el modelo anotador del conjunto de datos y un mismo
  modelo no debe generar las etiquetas y ser evaluado con ellas.
- **Chat en vivo (17).** Necesita una transmisión con el chat activo. Guarda solo el texto, la hora y el
  identificador de cada mensaje, sin el nombre del autor. Las salidas del notebook corresponden a la
  ejecución reportada en la monografía.
- **Comparación comercial (18).** Usa la API de pago de DeepSeek y guarda el avance en Google Drive para
  poder retomarlo; los tres modos costaron cerca de 1 USD en total (octubre de 2026). Las respuestas de un
  modelo comercial pueden variar entre ejecuciones aunque se use temperatura 0. Se publica sin salidas; los
  resultados están en la tabla final del notebook y en `final/comercial/` de Hugging Face.

## Cómo citar

```bibtex
@thesis{ortega2026emogol,
  author  = {Ortega Segura, Santiago Fernando},
  title   = {Inteligencia emocional en eventos futbol{\'i}sticos basada en lenguaje natural},
  school  = {Universidad del Cauca},
  address = {Popay{\'a}n, Colombia},
  type    = {Trabajo de grado},
  year    = {2026}
}
```

## Licencia

El código se publica bajo la licencia MIT (archivo `LICENSE`). Los datos y los adaptadores tienen sus propios
términos de uso, descritos en las tarjetas de sus repositorios en Hugging Face.
