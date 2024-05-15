import torch
import os
import csv
import pandas as pd
import numpy as np
import torch.nn as nn
from transformers import BertForSequenceClassification, BertConfig
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments, BertTokenizerFast, BertForMaskedLM, BertModel, BertConfig, AdamW

text_folder = "text_out_irs"
csv_direcotry = "csv_directory_test"
tokenizer_path = "my_pretrained_bert"
model_path = "pretrained_model"

# Example usage
text_files = [f for f in os.listdir(text_folder) if f.endswith('.txt')]

def load_embeddings(text_folder):
    irs = {}
    for file_name in text_files:
        with open(os.path.join(text_folder, file_name), 'r') as file:
            text = file.read()
            ir_name = os.path.splitext(file_name)[0]
            irs[ir_name] = text
    return irs


# for filename in os.listdir(text_folder):
#     if filename.endswith(".txt"):
#         embedding_name = os.path.splitext(filename)[0]
#         embedding_path = os.path.join(text_folder, filename)
#         embeddings[embedding_name] = torch.load(embedding_path)
# return embeddings

loaded_embeddings = load_embeddings(text_folder)
print("IR Loading done")

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

print("CSV Loading done")

X_embed =  [0 for _ in range(len(runtime_mapping))]
y =  [0 for _ in range(len(runtime_mapping))]

counter = 0

for rps in runtime_mapping:
    X_embed[counter] = loaded_embeddings[rps]
    y[counter] = runtime_mapping[rps]
    counter =  counter + 1

print("Mapping done")

# Load your pretrained tokenizer
tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path, model_max_length=256)

# Load your pretrained BERT model
#config = BertConfig.from_pretrained("path/to/your/pretrained/model/config")
model = BertForMaskedLM.from_pretrained(model_path,output_hidden_states=True)

# Freeze all the parameters of the BERT model
for param in model.parameters():
    param.requires_grad = False

# Define a custom regression head on top of BERT
class RegressionHead(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(RegressionHead, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Modify the BERT model for regression
bert_output_size = model.config.hidden_size
regression_head = RegressionHead(bert_output_size)
model.cls = regression_head


# Tokenize and encode the texts
encoded_texts = tokenizer(X_embed, padding=True, truncation=True, return_tensors="pt")

# Convert numbers to tensors
numbers_tensor = torch.tensor(y, dtype=torch.float32)

###############Newly added code######################

input_ids = encoded_texts['input_ids'].numpy()
attention_mask = encoded_texts['attention_mask'].numpy()

# Convert tensors to arrays
input_ids_array = input_ids.reshape(input_ids.shape[0], -1)
attention_mask_array = attention_mask.reshape(attention_mask.shape[0], -1)

# Split the dataset into train and validation sets
train_input_ids, val_input_ids, train_attention_mask, val_attention_mask, train_numbers, val_numbers = train_test_split(input_ids_array, attention_mask_array, numbers_tensor, test_size=0.2, random_state=42)

# Convert arrays back to tensors
train_input_ids_tensor = torch.tensor(train_input_ids)
val_input_ids_tensor = torch.tensor(val_input_ids)
train_attention_mask_tensor = torch.tensor(train_attention_mask)
val_attention_mask_tensor = torch.tensor(val_attention_mask)



# Create DataLoader for train and validation sets
train_dataset = TensorDataset(train_input_ids_tensor, train_attention_mask_tensor, train_numbers)
#train_dataset = train_dataset.squeeze()
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

val_dataset = TensorDataset(val_input_ids_tensor, val_attention_mask_tensor, val_numbers)
#val_dataset = val_dataset.squeeze()
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

###############Newly added code######################



# # Split the dataset into train and validation sets
# train_texts, val_texts, train_numbers, val_numbers = train_test_split(encoded_texts, numbers_tensor, test_size=0.2, random_state=42)

# # Split the dataset into train and validation sets
# train_texts = {'input_ids': encoded_texts['input_ids'][:len(train_numbers)],
#                'attention_mask': encoded_texts['attention_mask'][:len(train_numbers)]}
# val_texts = {'input_ids': encoded_texts['input_ids'][len(train_numbers):],
#              'attention_mask': encoded_texts['attention_mask'][len(train_numbers):]}


# # Create DataLoader for train and validation sets
# train_dataset = TensorDataset(train_texts['input_ids'], train_texts['attention_mask'], train_numbers)
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# val_dataset = TensorDataset(val_texts['input_ids'], val_texts['attention_mask'], val_numbers)
# val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for input_ids, attention_mask, targets in train_loader:
        #input_ids, attention_mask, targets = input_ids.to(device), attention_mask.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states
        # Assuming you want to use the last layer's hidden states
        last_layer_hidden_states = hidden_states[-1]
        # Take the [CLS] token representation (first token)
        cls_token_representation = last_layer_hidden_states[:, 0, :]
        predictions = model.cls(cls_token_representation)
        # predictions = outputs.last_hidden_state[:, 0, :]
        # predictions = model.cls(predictions)
        f_predictions = torch.flatten(predictions)
        f_targets = torch.flatten(targets)
        print(f_predictions)
        print(f_targets)
        loss = criterion(f_predictions, f_targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * input_ids.size(0)
    print("Evaluation")
    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for input_ids, attention_mask, targets in val_loader:
            #input_ids, attention_mask, targets = input_ids.to(device), attention_mask.to(device), targets.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.hidden_states
            last_layer_hidden_states = hidden_states[-1]  # Last layer's hidden states
            cls_token_representation = last_layer_hidden_states[:, 0, :]  # Take the [CLS] token representation
            predictions = model.cls(cls_token_representation)
            # predictions = outputs.last_hidden_state[:, 0, :]
            # predictions = model.cls(predictions)
            f_predictions = torch.flatten(predictions)
            f_targets = torch.flatten(targets)
            print(f_predictions)
            print(f_targets)
            loss = criterion(f_predictions, f_targets)
            val_loss += loss.item() * input_ids.size(0)
    
    train_loss /= len(train_loader.dataset)
    val_loss /= len(val_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Save the fine-tuned model
torch.save(model.state_dict(), "fine_tuned_bert_regression_model.pth")

# Prediction function
def predict(text):
    model.eval()
    with torch.no_grad():
        encoded_text = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        input_ids = encoded_text.input_ids
        attention_mask = encoded_text.attention_mask
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states
        last_layer_hidden_states = hidden_states[-1]  # Last layer's hidden states
        cls_token_representation = last_layer_hidden_states[:, 0, :]  # Take the [CLS] token representation
        predictions = model.cls(cls_token_representation)
        # predictions = outputs.last_hidden_state[:, 0, :]
        # predictions = model.cls(predictions)
        return predictions.item()
    
# Example usage
text_files = [f for f in os.listdir(text_folder) if f.endswith('.txt')]

predictions = {}
for file_name in text_files:
    with open(os.path.join(text_folder, file_name), 'r') as file:
        text = file.read()
        predicted_number = predict(text)
        predictions[file_name] = predicted_number
        print(f"Predicted number for '{file_name}': {predicted_number}")

csv_file = "output_rp_prediction.csv"

# Open the file in write mode
with open(csv_file, mode='w', newline='') as file:
    # Create a CSV writer
    writer = csv.writer(file)

    # Write the header if needed
    writer.writerow(['filename', 'runtime'])

    for file_name, prediction in predictions.items():
        writer.writerow([file_name, prediction])