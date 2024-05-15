import torch
import os
import pandas as pd
from transformers import BertForSequenceClassification, BertConfig
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
import numpy as np


embeddings_folder = "embeddings"
csv_direcotry = "csv_directory"

def load_embeddings(embeddings_folder):
    embeddings = {}
    for filename in os.listdir(embeddings_folder):
        if filename.endswith(".pth"):
            embedding_name = os.path.splitext(filename)[0]
            embedding_path = os.path.join(embeddings_folder, filename)
            embeddings[embedding_name] = torch.load(embedding_path)
    return embeddings

loaded_embeddings = load_embeddings(embeddings_folder)

counter = 0

# Step 2: Read Excel files and map runtimes
def read_excel_files(train_directory):
    runtime_mapping = {}
    for filename in os.listdir(train_directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(train_directory, filename)
            print(file_path)
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                runtime_mapping[row["filename"]] = row["runtime"]
    return runtime_mapping

runtime_mapping = read_excel_files(csv_direcotry)

X_embed =  [0 for _ in range(len(runtime_mapping))]
y =  [0 for _ in range(len(runtime_mapping))]

counter = 0

for rps in runtime_mapping:
    X_embed[counter] = loaded_embeddings[rps]
    y[counter] = runtime_mapping[rps]
    counter =  counter + 1

X_embed = torch.stack(X_embed)

# Convert data to PyTorch tensors
X_embed_tensor = torch.tensor(X_embed, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_embed_tensor, y_tensor, test_size=0.2, random_state=42)

# Create PyTorch datasets
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

# Define BERT model for regression
model = BertForSequenceClassification.from_pretrained('pretrained_model')  # num_labels=1 for regression

# Define training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=500,
    output_dir='finetuned_model_rp'
)

# Define a function to compute the mean squared error
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return {'mse': ((predictions - labels)**2).mean().item()}

# Instantiate the Trainer class
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Fine-tune the BERT model
trainer.train()

trainer.save_model('rp_finetuned_model')
