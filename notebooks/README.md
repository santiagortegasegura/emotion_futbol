# Notebooks

En orden de ejecución. Todos corren en Google Colab y leen y escriben sus datos en Hugging Face.

| Nº | Notebook | Qué hace | Nombre durante el desarrollo |
|---|---|---|---|
| 01 | `01_Preparacion_Datos.ipynb` | Limpieza y deduplicación (108.742 → 107.563) y muestra para etiquetado manual | `01_Preparacion_de_Datos` |
| 02 | `02_Refuerzo_Clase_Miedo.ipynb` | Refuerzo de la clase miedo en la muestra manual; pool de 106.923 | `01b_Enriquecimiento_Miedo` |
| 03 | `03_Etiquetado_Automatico.ipynb` | Calibración del anotador y etiquetado del pool (3 votos, confianza ≥ 0,67) | `02_Anotacion` |
| 04 | `04_Particion_Entrenamiento_Validacion.ipynb` | Partición estratificada: 72.000 entrenamiento y 18.000 validación | `04b_0_Preparar_Split_90k` |
| 05 | `05_Datos_Hugging_Face.ipynb` | Subida de los datos a Hugging Face | `05_migrar_datos_a_huggingface` |
| 06 | `06_Prueba_Final.ipynb` | Conjunto de prueba de 10.000 comentarios | `04c_Prueba_Final` |
| 07 | `07_Ajuste_Fino_Unsloth.ipynb` | Ajuste fino QLoRA de 8 modelos de lenguaje (ejecución: Llama 3.1 8B) | `06_entrenamiento_real_unsloth` |
| 08 | `08_Ajuste_Fino_Gemma.ipynb` | Ajuste fino QLoRA de Gemma-2 2B y 9B (ejecución: Gemma-2 2B, EmoGol-2B) | `06_entrenamiento_real_gemma_sin_unsloth` |
| 09 | `09_Evaluacion_Modelos_Lenguaje.ipynb` | Eficiencia: latencia, memoria y tamaño (ejecución: Mistral 7B) | `07_Evaluacion_Final_Split_Oficial` |
| 10 | `10_Metricas_Prueba_Final.ipynb` | Métricas de calidad oficiales sobre la prueba final (ejecución: Gemma-2 2B) | `07b_Evaluacion_Prueba_Final` |
| 11 | `11_Sensibilidad_Interpretacion.ipynb` | Verificación de la regla de interpretación de respuestas | `07c_Sensibilidad_Parseo` |
| 12 | `12_Modelos_Clasicos.ipynb` | SVM, árbol, Random Forest, CNN, BiLSTM y GNN | `08_Clasicos_Split_Oficial` |
| 13 | `13_BERT_RoBERTuito.ipynb` | RoBERTuito con LoRA | `09_BERT_Family_Split_Oficial` |
| 14 | `14_BERT_BETO.ipynb` | BETO con LoRA (mismo notebook que el 13) | `09_BERT_Family_Split_Oficial` |
| 15 | `15_Tabla_Maestra_Graficas.ipynb` | Tabla maestra de los 18 modelos y gráficas | `11_Consolidacion_Graficas` |
| 16 | `16_Prototipo_EmoGol.ipynb` | Prototipo EmoGol sobre un video de YouTube (final del Mundial 2026) | `10_Clasificador_Comentarios_Video` |

## Notas

- **Numeración.** Los notebooks se renumeraron en orden de ejecución para su publicación. Las salidas
  conservan los mensajes originales, así que en algunas aparece el nombre que tenían durante el desarrollo
  (columna de la derecha). Por la misma razón, los archivos de resultados en Hugging Face conservan sus
  prefijos originales: `07_resultados_finales.csv` (notebooks 09 y 10), `08_resultados_clasicos.csv`
  (notebook 12) y `09_resultados_bert_family.csv` (notebooks 13 y 14). Los archivos de datos
  `04b_entrenamiento_72k.csv` y `04b_validacion_interna_18k.csv` los genera el notebook 04.
- **Varias sesiones.** El etiquetado del pool (03) y los entrenamientos (07 y 08) se pueden retomar desde el
  último punto guardado, así que se hicieron en varias sesiones de Colab. De los entrenamientos se incluye
  la sesión en la que terminó un modelo por notebook.
- **Una ejecución por modelo.** Los notebooks 07, 08, 09 y 10 se corren una vez por modelo cambiando
  `CLAVE_MODELO`; aquí se incluye una ejecución de ejemplo de cada uno.
- **Resultados de los modelos de lenguaje.** La eficiencia (latencia, memoria, tamaño) viene del 09 y las
  métricas de calidad oficiales del 10. El 15 consolida los resultados de los 18 modelos.
