import json
import pandas as pd
import re

print("Cargando dataset...")
with open("data/raw/final_unified_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total original: {len(data):,}")

# Convertir a DataFrame
df = pd.DataFrame(data)

# ── LIMPIEZA ────────────────────────────────────────────────

# 1. Eliminar duplicados por texto
before = len(df)
df = df.drop_duplicates(subset=["text"])
print(f"Duplicados eliminados: {before - len(df):,}")

# 2. Eliminar textos muy cortos o muy largos
df = df[df["text"].str.len() >= 10]
df = df[df["text"].str.len() <= 500]
print(f"Textos fuera de rango eliminados")

# 3. Limpiar texto
def clean_text(text):
    text = str(text)
    # Eliminar URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Eliminar múltiples espacios
    text = re.sub(r'\s+', ' ', text)
    # Eliminar espacios al inicio y final
    text = text.strip()
    return text

df["text_clean"] = df["text"].apply(clean_text)

# 4. Eliminar textos vacíos después de limpieza
df = df[df["text_clean"].str.len() >= 10]

print(f"\nDataset limpio: {len(df):,} comentarios")
print(f"Reducción: {len(data) - len(df):,} comentarios eliminados")

# ── GUARDAR ─────────────────────────────────────────────────
df.to_csv("data/processed/my_dataset_clean.csv", index=False)
print("\n✅ Guardado en data/processed/my_dataset_clean.csv")
print("\nMuestra:")
print(df[["text_clean"]].head(3).to_string())
