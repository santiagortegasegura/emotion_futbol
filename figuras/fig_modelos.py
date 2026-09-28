from estilo_figuras import *

W, H = 6.5, 5.55
fig, ax = lienzo(W, H)
o = 0.3   # desplazamiento vertical para dejar espacio a la leyenda
carril(ax, 0.07, o + 0.08, W - 0.07, H - 0.05)

# 1 · Datos
dw, dh = 2.7, 0.85
dx, dy = (W - dw) / 2, o + 4.05
modulo(ax, dx, dy, dw, dh, "1", "Datos de entrenamiento",
       ["72.000 entrenamiento · 18.000 validación", "El mismo conjunto para las tres familias"])

# 2 · Familias
fw, fh = 1.8, 1.62
FX = [0.32, 2.35, 4.38]
fy = o + 1.95
modulo(ax, FX[0], fy, fw, fh, "2a", "Modelos de lenguaje (10)",
       ["Qwen2.5 1.5B · 3B · 7B", "Llama 3B · 8B · Mistral 7B", "Gemma-2 2B · 9B",
        "Phi-3 mini · Phi-3.5 mini", "QLoRA: 4 bits NF4 · LoRA r = 16", "3 épocas · lr 2×10⁻⁴ · lote 16"],
       salida="10 adaptadores LoRA")
modulo(ax, FX[1], fy, fw, fh, "2b", "Familia BERT (2)",
       ["RoBERTuito · BETO", "Cabeza de clasificación (7 clases)", "LoRA r = 8 (query y value)",
        "Pérdida ponderada por clase", "3 épocas"],
       salida="2 adaptadores LoRA")
modulo(ax, FX[2], fy, fw, fh, "2c", "Modelos clásicos (6)",
       ["SVM lineal (TF-IDF 1–2-gramas)", "Árbol y Random Forest", "(TF-IDF + SVD)",
        "CNN y BiLSTM (embeddings)", "GNN (TextGCN)", "Pesos balanceados por clase"],
       salida="6 modelos entrenados")

# 3 · Evaluación y selección
ew, eh = 1.85, 1.12
EX = [0.25, (W - ew) / 2, W - 0.25 - ew]
ey = o + 0.22
modulo(ax, EX[0], ey, ew, eh, "P", "Conjunto de prueba",
       ["10.000 comentarios", "No participa en el entrenamiento", "Igual para los 18 modelos"],
       estado="reservado")
modulo(ax, EX[1], ey, ew, eh, "3", "Evaluación común",
       ["Accuracy · precisión · recall · F1", "IC 95 % (bootstrap) · % inválidas", "Latencia · memoria · tamaño"],
       salida="Tabla comparativa (Cap. 5)")
modulo(ax, EX[2], ey, ew, eh, "4", "Selección del modelo",
       ["Gemma-2 2B + LoRA", "Accuracy 0,825 · F1 macro 0,776", "2,3 GB de memoria GPU"],
       salida="EmoGol-2B")

# Flechas: datos -> familias (abanico)
xc = W / 2
ytronco = o + 3.87
ax.plot([xc, xc], [dy, ytronco], color=FLECHA, lw=1.1, zorder=1)
ax.plot([FX[0] + fw / 2, FX[2] + fw / 2], [ytronco, ytronco], color=FLECHA, lw=1.1, zorder=1)
for x in FX:
    flecha(ax, (x + fw / 2, ytronco), (x + fw / 2, fy + fh + 0.12))

# familias -> evaluación (convergen)
yjunta = o + 1.77
for x in FX:
    ax.plot([x + fw / 2, x + fw / 2], [fy, yjunta], color=FLECHA, lw=1.1, zorder=1)
ax.plot([FX[0] + fw / 2, FX[2] + fw / 2], [yjunta, yjunta], color=FLECHA, lw=1.1, zorder=1)
flecha(ax, (xc, yjunta), (xc, ey + eh + 0.12))

flecha(ax, (EX[0] + ew, ey + eh / 2), (EX[1], ey + eh / 2), discontinua=True, color=GRIS)
flecha(ax, (EX[1] + ew, ey + eh / 2), (EX[2], ey + eh / 2))

leyenda(ax, 0.16, [("hecho", "Etapa del proceso"), ("reservado", "Reservado para la evaluación"),
                  ("linea", "Flujo")], x0=0.3, paso=2.0)
guardar(fig, "Figura_4_3_Entrenamiento_18_Modelos")
