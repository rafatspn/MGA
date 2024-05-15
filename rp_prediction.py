import torch
import os
import pandas as pd
from transformers import BertForSequenceClassification, BertConfig
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
import numpy as np
import csv

# Load the fine-tuned BERT model for prediction
model = BertForSequenceClassification.from_pretrained("pretrained_model")

embeddings_folder = "embeddings"

def load_embeddings(embeddings_folder):
    embeddings = {}
    for filename in os.listdir(embeddings_folder):
        if filename.endswith(".pth"):
            embedding_name = os.path.splitext(filename)[0]
            embedding_path = os.path.join(embeddings_folder, filename)
            embeddings[embedding_name] = torch.load(embedding_path)
    return embeddings

loaded_embeddings = load_embeddings(embeddings_folder)

# Define your input data for prediction
X_test = [0 for _ in range(len(loaded_embeddings))]  # Your input embeddings for testing (list or numpy array)
X_names = ['' for _ in range(len(loaded_embeddings))]

counter = 0

for emb in loaded_embeddings:
    X_test[counter] = loaded_embeddings[emb]
    X_names[counter] = emb
    counter =  counter + 1

# Convert test data to PyTorch tensor
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

predictions = {}
with torch.no_grad():
    model.eval()
    for file_name, batch in zip(X_names, DataLoader(X_test_tensor, batch_size=8)):
        outputs = model(batch)
        prediction = outputs.logits.squeeze().tolist()
        predictions[file_name] = prediction

# Now predictions dictionary will contain the filename as key and the regression output as value
# for file_name, prediction in predictions.items():
#     print(f"Prediction for {file_name} is {prediction}")

csv_file = "output_rp_prediction.csv"

# Open the file in write mode
with open(csv_file, mode='w', newline='') as file:
    # Create a CSV writer
    writer = csv.writer(file)

    # Write the header if needed
    writer.writerow(['filename', 'runtime'])

    for file_name, prediction in predictions.items():
        print(f"Prediction for {file_name} is {prediction}")
        writer.writerow([file_name, prediction])



