from estilo_figuras import *

W, H = 6.5, 5.05
fig, ax = lienzo(W, H)
o = 0.3
carril(ax, 0.07, o + 0.05, W - 0.07, H - 0.05)
BW, BH = 1.75, 1.2
X = [0.32, 2.32, 4.32]
y1, y2, y3 = o + 3.2, o + 1.68, o + 0.2

modulo(ax, X[0], y1, BW, BH, "1", "Entrada",
       ["Comentario de un video", "(primeras 72 h de publicado)", "Sin estado: cada texto se", "clasifica de forma independiente"])
modulo(ax, X[1], y1, BW, BH, "2", "Preprocesamiento",
       ["Filtro de validez", "Textos repetidos se cuentan una vez", "Recorte a 112 tokens",
        "Plantilla de prompt del entrenamiento"])
modulo(ax, X[2], y1, BW, BH, "3", "Inferencia EmoGol-2B",
       ["Gemma-2 2B (4 bits NF4) + LoRA", "Lotes de 16 textos", "Decodificación voraz (determinista)",
        "Máximo 6 tokens nuevos"])
modulo(ax, X[2], y2, BW, BH, "4", "Validación de salida",
       ["Se toma la primera palabra", "de la respuesta del modelo", "¿Es una de las 7 emociones?", "Sin reintentos"])
modulo(ax, X[1], y2, BW, BH, "5a", "Etiqueta de emoción",
       ["alegría · tristeza · enojo", "miedo · sorpresa · burla", "neutral"])
modulo(ax, X[2], y3, BW, BH, "5b", "Respuesta inválida",
       ["Se marca como «inválido»", "Se excluye del resumen", "Se conserva para su análisis"], estado="invalido")
modulo(ax, X[0], y2, BW, BH, "6", "Agregación y\nvisualización",
       ["Distribución de emociones", "Evolución por hora", "Ejemplos por emoción"])
modulo(ax, X[0], y3, BW, BH, "7", "Almacenamiento",
       ["Hugging Face Hub", "(prototipo/<caso>/)", "CSV · resumen JSON · gráficas"])

m1, m2, m3 = y1 + BH / 2, y2 + BH / 2, y3 + BH / 2
flecha(ax, (X[0] + BW, m1), (X[1], m1))
flecha(ax, (X[1] + BW, m1), (X[2], m1))
flecha(ax, (X[2] + BW / 2, y1), (X[2] + BW / 2, y2 + BH + 0.12))
flecha(ax, (X[2], m2), (X[1] + BW, m2), texto="Sí", dy=0.1)
flecha(ax, (X[2] + BW / 2, y2), (X[2] + BW / 2, y3 + BH + 0.12), texto="No", dx=0.12, color=ROJO)
flecha(ax, (X[1], m2), (X[0] + BW, m2))
flecha(ax, (X[0] + BW / 2, y2), (X[0] + BW / 2, y3 + BH + 0.12))
flecha(ax, (X[2], m3), (X[0] + BW, m3), discontinua=True, color=ROJO, texto="se guarda marcado", dy=0.1)

leyenda(ax, 0.14, [("hecho", "Etapa del flujo"), ("invalido", "Salida no válida"),
                   ("linea", "Flujo"), ("linea_punteada", "Registro")], x0=0.3, paso=1.55)
guardar(fig, "Figura_4_4_Flujo_Inferencia")
