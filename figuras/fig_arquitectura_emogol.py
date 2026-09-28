"""Figura: arquitectura general de EmoGol (estilo de las Figuras 4.1-4.3).

Se dibuja al tamaño real de la página (6,5 in de ancho) para que el texto se lea en el documento.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, FancyArrowPatch

AZUL = "#1f3864"      # bordes y títulos de módulos implementados
FLECHA = "#2e5395"
ROJO = "#7b1c2c"      # planeado / no implementado
GRIS_TXT = "#333333"
FONDO_CARRIL = "#f3f5f9"

plt.rcParams["font.family"] = "DejaVu Sans"

W, H = 6.5, 7.2
fig = plt.figure(figsize=(W, H), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

BW, BH = 1.75, 1.0          # tamaño de cada módulo
X = [0.32, 2.32, 4.32]         # columnas
R = 0.02                     # redondeo


def carril(y0, y1, titulo, subtitulo):
    ax.add_patch(FancyBboxPatch((0.07, y0), W - 0.14, y1 - y0, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=FONDO_CARRIL, ec="none", zorder=0))
    ax.text(0.25, y1 - 0.2, titulo, fontsize=8.5, fontweight="bold", color=AZUL, va="center")
    ax.text(0.25, y1 - 0.38, subtitulo, fontsize=6.3, color=GRIS_TXT, va="center", style="italic")


def modulo(x, y, etiqueta, titulo, lineas, planeado=False, planeadas=()):
    """planeadas: líneas de alcance futuro, dibujadas dentro de un recuadro rojo punteado."""
    color = ROJO if planeado else AZUL
    ax.add_patch(FancyBboxPatch((x, y), BW, BH, boxstyle=f"round,pad=0,rounding_size={R*4}",
                                fc="white", ec=color, lw=1.2, ls="--" if planeado else "-", zorder=2))
    ax.add_patch(Ellipse((x, y + BH), 0.34, 0.2, fc="white", ec=color, lw=1.1, zorder=3))
    ax.text(x, y + BH, etiqueta, ha="center", va="center", fontsize=6.3, fontweight="bold", color=color, zorder=4)
    ax.text(x + BW / 2, y + BH - 0.22, titulo, ha="center", va="center", fontsize=7.4,
            fontweight="bold", color="#111111", zorder=4)
    for i, linea in enumerate(lineas):
        ax.text(x + BW / 2, y + BH - 0.47 - i * 0.16, linea, ha="center", va="center",
                fontsize=5.9, color=GRIS_TXT, zorder=4)
    for j, linea in enumerate(planeadas):
        yl = y + BH - 0.47 - (len(lineas) + j) * 0.16 - 0.02
        ax.add_patch(FancyBboxPatch((x + 0.1, yl - 0.075), BW - 0.2, 0.15,
                                    boxstyle="round,pad=0,rounding_size=0.03", fc="white", ec=ROJO,
                                    lw=0.9, ls=(0, (3, 2)), zorder=3))
        ax.text(x + BW / 2, yl, linea, ha="center", va="center", fontsize=5.9, color=ROJO,
                style="italic", zorder=4)


def flecha(p0, p1, discontinua=False, color=FLECHA, estilo="arc3"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, lw=1.1, color=color,
                                 ls=(0, (3, 2)) if discontinua else "-", connectionstyle=estilo, zorder=1))


# ---------------- Carril A: construcción del modelo ----------------
yA = 5.3
carril(5.05, 7.1, "A · Construcción del modelo (fuera de línea)",
       "Se realiza una sola vez; sus detalles están en las Figuras 4.2 y 4.3 y en el Capítulo 5.")
modulo(X[0], yA, "A1", "Conjunto de datos EmoGol",
       ["100.000 comentarios · 7 emociones", "72k entren. · 18k valid. · 10k prueba", "(Fig. 4.2)"])
modulo(X[1], yA, "A2", "Ajuste fino QLoRA",
       ["10 modelos de lenguaje comparados", "4 bits NF4 · LoRA r = 16", "(Fig. 4.3)"])
modulo(X[2], yA, "A3", "Modelo EmoGol-2B",
       ["Gemma-2 2B + adaptador LoRA", "Accuracy 0,825 · F1 macro 0,776", "Publicado en Hugging Face Hub"])
flecha((X[0] + BW, yA + BH / 2), (X[1], yA + BH / 2))
flecha((X[1] + BW, yA + BH / 2), (X[2], yA + BH / 2))

# ---------------- Carril B: prototipo en línea ----------------
carril(1.0, 4.93, "B · Prototipo EmoGol (en línea)",
       "Implementado en Google Colab con GPU T4 (videos grabados, notebook 10).")
y1, y2 = 3.2, 1.62
modulo(X[0], y1, "B1", "Fuente: YouTube",
       ["Videos de partidos (comentarios)", "YouTube Data API v3"],
       planeadas=["Chat en vivo (planeado)"])
modulo(X[1], y1, "B2", "Ingesta",
       ["Video: primeras 72 h, máx. 3.000", "Muestreo aleatorio (semilla 42)"],
       planeadas=["En vivo: cada 10 s (planeado)"])
modulo(X[2], y1, "B3", "Preprocesamiento",
       ["Filtro de validez", "Textos repetidos se cuentan una vez", "Recorte a 112 tokens + prompt"])
modulo(X[2], y2, "B4", "Clasificación EmoGol-2B",
       ["Lotes de 16 · decodificación voraz", "Salida validada: 7 emociones", "o «inválido»"])
modulo(X[1], y2, "B5", "Agregación y visualización",
       ["Distribución de emociones", "Evolución por hora", "Ejemplos por emoción"])
modulo(X[0], y2, "B6", "Almacenamiento",
       ["Hugging Face Hub", "(prototipo/<caso>/)", "CSV · resumen JSON · gráficas"])

flecha((X[0] + BW, y1 + BH / 2), (X[1], y1 + BH / 2))
flecha((X[1] + BW, y1 + BH / 2), (X[2], y1 + BH / 2))
flecha((X[2] + BW / 2, y1), (X[2] + BW / 2, y2 + BH + 0.1))
flecha((X[2], y2 + BH / 2), (X[1] + BW, y2 + BH / 2))
flecha((X[1], y2 + BH / 2), (X[0] + BW, y2 + BH / 2))

# El modelo publicado (A3) se carga en el clasificador (B4): ruta por el margen derecho
xr = X[2] + BW + 0.22
ax.plot([X[2] + BW, xr, xr], [yA + BH / 2, yA + BH / 2, y2 + BH / 2 + 0.12], color=FLECHA, lw=1.1,
        ls=(0, (3, 2)), zorder=1)
flecha((xr, y2 + BH / 2 + 0.12), (X[2] + BW, y2 + BH / 2 + 0.12), discontinua=True)
ax.text(xr + 0.07, (yA + y2 + BH) / 2, "carga del adaptador", rotation=90, ha="center", va="center",
        fontsize=6, color=FLECHA, style="italic")

# Servicio en la nube planeado: envuelve B2-B5
pad = 0.1
ax.add_patch(FancyBboxPatch((X[1] - pad - 0.06, y2 - pad - 0.28), X[2] + BW - X[1] + 2 * pad + 0.06,
                            (y1 + BH + 0.14) - (y2 - pad - 0.28), boxstyle="round,pad=0,rounding_size=0.06",
                            fc="none", ec=ROJO, lw=1.1, ls=(0, (4, 2.5)), zorder=1))
ax.text(X[1] + (X[2] + BW - X[1]) / 2, y2 - pad - 0.15,
        "Servicio EmoGol en la nube (planeado): API + página de demostración",
        ha="center", va="center", fontsize=6.3, color=ROJO, fontweight="bold")

# ---------------- Leyenda ----------------
ly = 0.72
ax.add_patch(FancyBboxPatch((0.25, ly - 0.07), 0.3, 0.14, boxstyle="round,pad=0,rounding_size=0.02",
                            fc="white", ec=AZUL, lw=1.1))
ax.text(0.62, ly, "Implementado", fontsize=6.2, va="center", color=GRIS_TXT)
ax.add_patch(FancyBboxPatch((1.6, ly - 0.07), 0.3, 0.14, boxstyle="round,pad=0,rounding_size=0.02",
                            fc="white", ec=ROJO, lw=1.1, ls=(0, (3, 2))))
ax.text(1.97, ly, "Planeado", fontsize=6.2, va="center", color=GRIS_TXT)
ax.plot([2.85, 3.2], [ly, ly], color=FLECHA, lw=1.1)
ax.text(3.27, ly, "Flujo de datos", fontsize=6.2, va="center", color=GRIS_TXT)
ax.plot([4.25, 4.6], [ly, ly], color=FLECHA, lw=1.1, ls=(0, (3, 2)))
ax.text(4.67, ly, "Carga del modelo", fontsize=6.2, va="center", color=GRIS_TXT)

fig.savefig("Figura_4_1_Arquitectura_EmoGol.png", dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.04)
fig.savefig("Figura_4_1_Arquitectura_EmoGol.pdf", facecolor="white", bbox_inches="tight", pad_inches=0.04)
print("ok")
