import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertModel
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

input_irs = './input_irs'
text_inp_irs = './text_in_irs'
gen_irs = './output_irs'
text_gen_irs = './text_out_irs'
embedding_path='./embeddings'

def ir_to_text(ll_filename, output_directory):
    with open(ll_filename, 'r') as ll_file:
        ll_content = ll_file.read()

    txt_filename = os.path.join(output_directory, os.path.basename(ll_filename).replace('.ll', '.txt'))

    with open(txt_filename, 'w') as txt_file:
        txt_file.write(ll_content)

    print(f"Conversion successful: {os.path.basename(ll_filename)} -> {os.path.basename(txt_filename)}")

def read_irs(directory_path, output_directory):
    all_files = os.listdir(directory_path)

    ll_files = [file for file in all_files if file.endswith('.ll')]

    os.makedirs(output_directory, exist_ok=True)

    for ll_file in ll_files:
        ll_file_path = os.path.join(directory_path, ll_file)
        ir_to_text(ll_file_path, output_directory)

# input_directory = input_irs

# output_directory = text_inp_irs

read_irs(input_irs, text_inp_irs)

# Define a custom dataset for unsupervised text data
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return encoding

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Example unsupervised text data (replace with your own data)
# texts = [
#     "This is the first text.",
#     "Another example text.",
#     "More unsupervised data."
# ]

text_files = [f for f in os.listdir(text_inp_irs) if f.endswith('.txt')]

texts = []
for file_name in text_files:
    with open(os.path.join(text_inp_irs, file_name), 'r') as file:
        text = file.read()
        texts.append(text)

################NEWLY ADDED START#################

input_ids = []
attention_masks = []

#TOKENIZATION PROCESS 1
# max_length=768
for ir_text in texts:
    encoded_ir = tokenizer.encode_plus(ir_text, truncation=True, padding='max_length', add_special_tokens=True, max_length=512,  return_attention_mask=True)
    input_ids.append(encoded_ir['input_ids'])
    attention_masks.append(encoded_ir['attention_mask'])

# #find max len
# max_len=0
# for i in range(len(input_ids)):
#     if len(input_ids[i])> max_len:
#         max_len = len(input_ids[i])

#Padding
# for i in range(len(input_ids)):
#     padding_len = max_len - len(input_ids[i])
#     for j in range(padding_len):
#         input_ids[i].append(0)
#         attention_masks[i].append(0)


#print
for i in range(len(input_ids)):
    print(len(input_ids[i]))
    print(len(attention_masks[i]))



#TOKENIZATION PROCESS 2
# encoded_ir = tokenizer.encode_plus(texts, add_special_tokens=True, max_length=768, padding=True, return_attention_mask=True, batch=True)
# print(encoded_ir)
# input_ids = encoded_ir['input_ids']
# attention_masks = encoded_ir['attention_mask']

#print(input_ids[0])

input_ids = torch.tensor(input_ids)
attention_masks = torch.tensor(attention_masks)

train_dataset = TensorDataset(input_ids, attention_masks)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)


################NEWLY ADDED END#################

# Fine-tune the model (you can replace this with your own fine-tuning process)
# train_dataset = TextDataset(texts, tokenizer, max_length=768)
# train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

#print(train_dataset)

optimizer = AdamW(model.parameters(), lr=1e-5)
for epoch in range(10):
    model.train()
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        optimizer.zero_grad()
        #print(batch)
        inputs,attention_mask = batch
        #batch['input_ids']
        #attention_mask = batch['attention_mask']
        #inputs = inputs.squeeze(0)
        outputs = model(inputs, attention_mask=attention_mask)
        # print(outputs)
        # loss = outputs.loss
        # loss.backward()
        # optimizer.step()

# Save the fine-tuned model
#model.save_pretrained("finetuned_bert")
torch.save(model.state_dict(),'trained_bert.pth')

# Load the fine-tuned model
#loaded_model = BertForSequenceClassification.from_pretrained("finetuned_bert")
loaded_model = BertModel.from_pretrained('bert-base-uncased')
loaded_model.load_state_dict(torch.load('trained_bert.pth'))# Example new text files (replace with your own filenames)

read_irs(gen_irs, text_gen_irs)

new_files = [f for f in os.listdir(text_gen_irs) if f.endswith('.txt')]
#new_files = ["file1.txt"]

# Generate embeddings for new text files
for filename in new_files:
    with open(os.path.join(text_gen_irs,filename), 'r') as file:
        text = file.read()
        encoding = tokenizer(text,add_special_tokens=True, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        # print("PRINTING LEN")
        # print(len(encoding))
        output = loaded_model(**encoding)
        embeddings = output.last_hidden_state.mean(dim=1)  # Use mean pooling for sentence embeddings
        new_emb_path = os.path.join(embedding_path, os.path.basename(filename).replace('.txt', '.pth'))
        torch.save(embeddings, new_emb_path)
        #print(f"Embeddings for {filename}: {embeddings.tolist()}")
