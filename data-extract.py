import os
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import random

def parquet_files_in_dir(directory):
    files = []
    for filename in os.listdir(directory):
        # get all files ending with .parquet and append them to the files list
        if filename.endswith(".parquet") and os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)
    return files

def process_file(args):

    directory, filename, output_file = args
    # get file path for each parquet file in folder
    file_path = os.path.join(directory, filename)

    df = pd.read_parquet(file_path)

    # drop any empty or corrupt data, turn into strings and put into list
    text_data = df['text'].dropna().astype(str).tolist()
    text = '\n'.join(text_data) + '\n'

    # append cleaned data into output file
    with open(output_file, "a", encoding="utf-8") as outfile:
        outfile.write(text)

    # get a set of unique chars
    characters = set(text)
    return characters

def process_files_in_parallel(files, folder_path, output_file):
    # Create master vocab list
    vocab = set()

    # setup cpu cores
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        # Package instructions for each core
        args = [(folder_path, filename, output_file) for filename in files]

        # use tqdm to track the progress of the cores
        for characters in tqdm(executor.map(process_file, args), total=len(files)):
            # add discovered chars to master vocab list
            vocab.update(characters)
    
    return vocab


folder_path = "/home/yurus/Dev/github.com/YuriSorice/llm-basics/local_openwebtext/plain_text"
output_file_train = "output_train.txt"
output_file_val = "output_val.txt"
vocab_file = "vocab.txt"

# master list of parquet files
files = parquet_files_in_dir(folder_path)
total_files = len(files)
print(f"Found {total_files} total Parquet files.")

# 90/10 training to validation split
split_index = int(total_files * 0.9)
files_train = files[:split_index] # 90% to training
files_val = files[split_index:] # 10% to validation

# Sample rate
sample_rate = 0.01
files_train_sampled = random.sample(files_train, max(1, int(len(files_train) * sample_rate)))
files_val_sampled = random.sample(files_val, max(1, int(len(files_val) * sample_rate)))

print(f"Procesing {len(files_train_sampled)} training files and {len(files_val_sampled)} validation files")

# clear in case of re-run of script
open(output_file_train, 'w').close()
open(output_file_val, 'w').close()

# begin processing files

print("Extracting training data...")
vocab_train = process_files_in_parallel(files_train_sampled, folder_path, output_file_train)

print("Extracting validation data...")
vocab_val = process_files_in_parallel(files_val_sampled, folder_path, output_file_val)

print("Saving Master Vocabulary...")
vocab = vocab_train.union(vocab_val)
with open(vocab_file, "w", encoding="utf-8") as vfile:
    for char in sorted(vocab):
        vfile.write(char + '\n')

print("Data Extraction Complete.")