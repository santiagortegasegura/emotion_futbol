from estilo_figuras import *

W, H = 6.5, 4.3
fig, ax = lienzo(W, H)
BW, BH = 1.75, 1.3
X = [0.32, 2.32, 4.32]
y1, y2 = 2.55, 0.6
carril(ax, 0.07, 0.35, W - 0.07, H - 0.05)

modulo(ax, X[0], y1, BW, BH, "1", "Extracción de\ncomentarios",
       ["Script propio · YouTube Data API v3", "178 videos (2022 – mayo 2026)", "Filtro: mínimo 10 caracteres"],
       salida="108.742 comentarios")
modulo(ax, X[1], y1, BW, BH, "2", "Limpieza y\ndeduplicación",
       ["Normalización del texto", "Eliminación de duplicados exactos"],
       salida="107.563 comentarios")
modulo(ax, X[2], y1, BW, BH, "3", "Anotación asistida\npor LLM",
       ["Qwen2.5-7B vía Ollama", "Few-shot + reglas · 3 votos", "Calibrada con 640 etiquetas humanas"],
       salida="106.923 etiquetados")
modulo(ax, X[2], y2, BW, BH, "4", "Filtrado por\nconfianza",
       ["Confianza ≥ 0,67", "(al menos 2 de 3 votos iguales)", "Etiqueta dentro de las 7 emociones"],
       salida="100.858 comentarios")
modulo(ax, X[1], y2, BW, BH, "5", "Selección y\npartición",
       ["Aleatoria y estratificada (semilla 42)", "72.000 entren. · 18.000 valid.", "10.000 prueba (sin repetidos)"],
       salida="100.000 comentarios")
modulo(ax, X[0], y2, BW, BH, "6", "Publicación en\nHugging Face Hub",
       ["Repositorio de datos privado", "Mismos datos para", "los 18 modelos"],
       salida="Conjunto de datos EmoGol")

ym1, ym2 = y1 + BH / 2, y2 + BH / 2
flecha(ax, (X[0] + BW, ym1), (X[1], ym1))
flecha(ax, (X[1] + BW, ym1), (X[2], ym1))
flecha(ax, (X[2] + BW / 2, y1), (X[2] + BW / 2, y2 + BH + 0.12))
flecha(ax, (X[2], ym2), (X[1] + BW, ym2))
flecha(ax, (X[1], ym2), (X[0] + BW, ym2))

guardar(fig, "Figura_4_2_Construccion_Datos")
