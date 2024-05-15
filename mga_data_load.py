import os 
import json

text_inp_irs = './text_in_irs'


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


json_string = json.dumps(texts)

# Open a file in write mode
with open('ir_array.json', 'w') as f:
    # Write the JSON string to the file
    f.write(json_string)

# Close the file
f.close()

print("DONE")