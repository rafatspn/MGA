from datasets import *
from transformers import *
from tokenizers import *
import os
import json
import torch
from itertools import chain
from datasets import list_datasets
import pandas as pd
#from transformers import BertModel, BertConfig

text_inp_irs = './text_in_irs'
gen_irs = './output_irs'
text_gen_irs = './text_out_irs'
embedding_path='./embeddings'


#########Dataset preparation online start ###########

# dataset_list = list_datasets()

# print(len(dataset_list))
# print(', '.join(dataset for dataset in dataset_list))
# print(dataset_list[:9])

# dataset = load_dataset('cc_news',split='train')

#########Dataset preparation online end ###########

text_files = [f for f in os.listdir(text_inp_irs) if f.endswith('.txt')]

texts = []
for file_name in text_files:
    with open(os.path.join(text_inp_irs, file_name), 'r') as file:
        text = file.read()
        texts.append({'text': text})

dataset = Dataset.from_pandas(pd.DataFrame(data=texts))

d = dataset.train_test_split(train_size=1, test_size=1)
d['train'], d['test']

def dataset_to_text(dataset, output_filename='data.txt'):
  with open(output_filename, "w") as f:
    for t in dataset['text']:
      print(t,file=f)

dataset_to_text(d['train'],'train.txt')
dataset_to_text(d['test'],'test.txt')

special_tokens = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"
]

files = ["train.txt"]

vocab_size = 20_000

max_length = 256

truncate_longer_samples = True

tokenizer = BertWordPieceTokenizer()

tokenizer.train(files=files, vocab_size=vocab_size, special_tokens=special_tokens)

tokenizer.enable_truncation(max_length=max_length)


my_model_path = 'my_pretrained_bert'

if not os.path.isdir(my_model_path):
  os.mkdir(my_model_path)

tokenizer.save_model(my_model_path)

with open(os.path.join(my_model_path,'config.json'),"w") as f:
  tokenizer_cfg = {
      "do_lower_case": True,
      "unk_token": "[UNK]",
      "sep_token": "[SEP]",
      "pad_token": "[PAD]",
      "cls_token": "[CLS]",
      "mask_token":"[MASK]",
      "model_max_length":max_length,
      "max_len": max_length
  }
  json.dump(tokenizer_cfg, f)

tokenizer = BertTokenizerFast.from_pretrained(my_model_path)

def encode_with_truncation(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length, return_special_tokens_mask=True)

def encode_without_truncation(examples):
  return tokenizer(examples["text"], return_special_tokens_mask=True)

encode = encode_with_truncation if truncate_longer_samples else encode_without_truncation

train_dataset = d["train"].map(encode, batched=True)
test_dataset = d["test"].map(encode, batched=True)


if truncate_longer_samples:
  train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
  test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
else:
  train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "special_tokens_mask"])
  test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "special_tokens_mask"])


def group_texts(examples):
  concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
  total_length = len(concatenated_examples[list(examples.keys())])

  if total_length >= max_length:
    total_length = (total_length // max_length) * max_length

  result = {
    k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
    for k, t in concatenated_examples.items()
  }
  return result


if not truncate_longer_samples:
  train_dataset = train_dataset.map(group_texts, batched=True,
                                    desc=f"Grouping texts in chunks of {max_length}")
  test_dataset = test_dataset.map(group_texts, batched=True,
                                  desc=f"Grouping texts in chunks of {max_length}")
  # convert them from lists to torch tensors
  train_dataset.set_format("torch")
  test_dataset.set_format("torch")

len(train_dataset), len(test_dataset)

train_dataset

train_dataset[0]

model_config=BertConfig(vocab_size=vocab_size,max_position_embeddings=max_length)

configuration = BertConfig()

model = BertModel(configuration)

configuration = model.config

model_config

model = BertForMaskedLM(config=model_config)

model

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=True,mlm_probability=0.2, return_tensors='pt')

data_collator

training_args=TrainingArguments(
    output_dir=my_model_path,
    evaluation_strategy='steps',
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=64,
    logging_steps=10,
    save_steps=100,
    load_best_model_at_end=True
)

training_args

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

trainer.save_model('pretrained_model')