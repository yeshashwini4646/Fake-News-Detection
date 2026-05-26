import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load dataset
data = pd.read_csv("dataset/processed.csv")

# Clean dataset
data = data.dropna()
data["text"] = data["text"].astype(str)

texts = data["text"].tolist()
labels = data["label"].tolist()

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

encodings = tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt"
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    encodings['input_ids'],
    labels,
    test_size=0.2,
    random_state=42
)

# Load BERT model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

print("BERT Model Loaded Successfully")

# Save model
model.save_pretrained("model/")
tokenizer.save_pretrained("model/")

print("Model Saved Successfully")