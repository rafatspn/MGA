import os
import torch
from transformers import *
from tokenizers import *

tokenizer_path = 'my_pretrained_bert'
model_path = 'pretrained_model'
gen_irs = 'output_irs'
text_gen_irs = 'text_out_irs'
embedding_path = 'embeddings'
max_length = 256
truncate_longer_samples = False

tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

model = BertModel.from_pretrained(model_path)

# def encode_with_truncation(examples):
#     return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length, return_special_tokens_mask=True)

# def encode_without_truncation(examples):
#   return tokenizer(examples["text"], return_special_tokens_mask=True)

#encode = encode_with_truncation if truncate_longer_samples else encode_without_truncation

text_files = [f for f in os.listdir(text_gen_irs) if f.endswith('.txt')]

# Generate embeddings for new text files
for filename in text_files:
    with open(os.path.join(text_gen_irs,filename), 'r') as file:
        text = file.read()
        encoding = tokenizer(text, truncation=True, padding="max_length", max_length=max_length, return_tensors='pt')
        output = model(**encoding)
        embeddings = output.last_hidden_state.mean(dim=1)  # Use mean pooling for sentence embeddings
        new_emb_path = os.path.join(embedding_path, os.path.basename(filename).replace('.txt', '.pth'))
        torch.save(embeddings, new_emb_path)
        #print(f"Embeddings for {filename}: {embeddings.tolist()}")

