"""Estilo común para las figuras del Capítulo 4 de EmoGol.

Todas las figuras se dibujan al ancho real de la página (6,5 in ≈ 16,5 cm) para que el texto se lea
en el documento sin reescalar. Los títulos NO van dentro de la imagen: la leyenda se pone en Word.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, FancyArrowPatch
from PIL import Image, ImageChops

AZUL = "#1f3864"
FLECHA = "#2e5395"
ROJO = "#7b1c2c"
GRIS = "#6f6f6f"
GRIS_TXT = "#333333"
FONDO_CARRIL = "#f3f5f9"

plt.rcParams["font.family"] = "DejaVu Sans"


def lienzo(ancho, alto):
    fig = plt.figure(figsize=(ancho, alto), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, ancho); ax.set_ylim(0, alto); ax.axis("off")
    return fig, ax


def carril(ax, x0, y0, x1, y1, titulo=None, subtitulo=None):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=FONDO_CARRIL, ec="none", zorder=0))
    if titulo:
        ax.text(x0 + 0.15, y1 - 0.2, titulo, fontsize=8.5, fontweight="bold", color=AZUL, va="center")
    if subtitulo:
        ax.text(x0 + 0.15, y1 - 0.38, subtitulo, fontsize=6.3, color=GRIS_TXT, va="center", style="italic")


def modulo(ax, x, y, w, h, etiqueta, titulo, lineas=(), estado="hecho", tam_titulo=7.4, salida=None):
    """estado: 'hecho' (azul, continuo), 'planeado' (rojo, punteado), 'reservado' (gris, punteado),
    'invalido' (rojo, continuo)."""
    color = {"hecho": AZUL, "planeado": ROJO, "reservado": GRIS, "invalido": ROJO}[estado]
    punteado = estado in ("planeado", "reservado")
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                fc="white", ec=color, lw=1.2, ls=(0, (4, 2.5)) if punteado else "-", zorder=2))
    if etiqueta:
        ax.add_patch(Ellipse((x, y + h), 0.34, 0.2, fc="white", ec=color, lw=1.1, zorder=3))
        ax.text(x, y + h, etiqueta, ha="center", va="center", fontsize=6.3, fontweight="bold",
                color=color, zorder=4)
    titulos = titulo.split("\n")
    for i, t in enumerate(titulos):
        ax.text(x + w / 2, y + h - 0.22 - i * 0.15, t, ha="center", va="center", fontsize=tam_titulo,
                fontweight="bold", color="#111111", zorder=4)
    base = y + h - 0.22 - (len(titulos) - 1) * 0.15 - 0.25
    for i, linea in enumerate(lineas):
        ax.text(x + w / 2, base - i * 0.155, linea, ha="center", va="center", fontsize=5.9,
                color=GRIS_TXT, zorder=4)
    if salida:
        ax.plot([x + 0.12, x + w - 0.12], [y + 0.27, y + 0.27], color="#d5dbe7", lw=0.6, zorder=3)
        ax.text(x + w / 2, y + 0.14, salida, ha="center", va="center", fontsize=6.4, fontweight="bold",
                color=FLECHA, zorder=4)


def flecha(ax, p0, p1, discontinua=False, color=FLECHA, estilo="arc3", texto=None, dx=0.0, dy=0.0,
           rot=0, ha="center"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, lw=1.1, color=color,
                                 ls=(0, (3, 2)) if discontinua else "-", connectionstyle=estilo,
                                 shrinkA=0, shrinkB=0, zorder=1))
    if texto:
        xm, ym = (p0[0] + p1[0]) / 2 + dx, (p0[1] + p1[1]) / 2 + dy
        ax.text(xm, ym, texto, ha=ha, va="center", fontsize=5.9, color=color, style="italic",
                rotation=rot, zorder=5, bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))


def poligonal(ax, puntos, discontinua=False, color=FLECHA):
    """Línea con quiebres que termina en flecha."""
    xs, ys = zip(*puntos[:-1])
    ax.plot(list(xs) + [puntos[-2][0]], list(ys) + [puntos[-2][1]], color=color, lw=1.1,
            ls=(0, (3, 2)) if discontinua else "-", zorder=1)
    flecha(ax, puntos[-2], puntos[-1], discontinua=discontinua, color=color)


def leyenda(ax, y, items, x0=0.3, paso=1.55):
    x = x0
    for tipo, texto in items:
        if tipo in ("hecho", "planeado", "reservado", "invalido"):
            color = {"hecho": AZUL, "planeado": ROJO, "reservado": GRIS, "invalido": ROJO}[tipo]
            ax.add_patch(FancyBboxPatch((x, y - 0.07), 0.3, 0.14, boxstyle="round,pad=0,rounding_size=0.02",
                                        fc="white", ec=color, lw=1.1,
                                        ls=(0, (3, 2)) if tipo in ("planeado", "reservado") else "-"))
        else:
            ax.plot([x, x + 0.3], [y, y], color=FLECHA, lw=1.1, ls=(0, (3, 2)) if tipo == "linea_punteada" else "-")
        ax.text(x + 0.37, y, texto, fontsize=6.2, va="center", color=GRIS_TXT)
        x += paso


def guardar(fig, nombre):
    fig.savefig(f"{nombre}.pdf", facecolor="white", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(f"{nombre}.png", dpi=300, facecolor="white")
    im = Image.open(f"{nombre}.png").convert("RGB")
    caja = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
    m = 40
    im.crop((max(caja[0] - m, 0), max(caja[1] - m, 0), min(caja[2] + m, im.size[0]),
             min(caja[3] + m, im.size[1]))).save(f"{nombre}.png", dpi=(300, 300))
    plt.close(fig)
    print(f"Guardada: {nombre}")
