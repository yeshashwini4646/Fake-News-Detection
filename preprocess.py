import pandas as pd
import re

def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

# Load datasets
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

# Labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
data = pd.concat([fake, true])

# Remove missing values
data = data.dropna()

# Convert text column to string
data["text"] = data["text"].astype(str)

# Clean text
data["text"] = data["text"].apply(clean_text)

# Save processed dataset
data.to_csv("dataset/processed.csv", index=False)

print("Preprocessing Completed Successfully")