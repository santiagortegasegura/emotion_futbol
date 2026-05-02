from datasets import load_dataset
import pandas as pd
from langdetect import detect

print("Downloading EmoEvent dataset ...")
dataset = load_dataset("SINAI/EmoEvent")

print("Dataset downloaded")


# Convert to pandas dataframe
train_df = pd.DataFrame(dataset["train"])
val_df = pd.DataFrame(dataset["validation"])
test_df = pd.DataFrame(dataset["test"])

# Function to detect language
def is_spanish(text):
	try:
		return detect(str(text)) == "es"
	except:
		return False

print("\nFiltering Spanish tweets...")
train_es = train_df[train_df["tweet"].apply(is_spanish)]
val_es = val_df[val_df["tweet"].apply(is_spanish)]
test_es = test_df[test_df["tweet"].apply(is_spanish)]

print(f"Train: {len(train_df)} → {len(train_es)} Spanish tweets")
print(f"Validation: {len(val_df)} → {len(val_es)} Spanish tweets")
print(f"Test: {len(test_df)} → {len(test_es)} Spanish tweets")

print("\nEmotion distribution (Spanish only):")
print(train_es["emotion"].value_counts())

# Save to CSV
train_es.to_csv("data/raw/emoevent_train.csv", index=False)
val_es.to_csv("data/raw/emoevent_val.csv", index=False)
test_es.to_csv("data/raw/emoevent_test.csv", index=False)

print("\nSaved to data/raw/")
