import os

input_dir = './input_irs'
output_dir = './text_in_irs'

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

read_irs(input_dir, output_dir)

