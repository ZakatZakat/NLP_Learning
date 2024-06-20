from transformers import BertTokenizerFast
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import urllib.request

def read_csv_default(data=None, Target=None):
        # Try with ISO-8859-1
    try:
        data = pd.read_csv('/home/oskar/NLP/datasets/Corona_NLP_train.csv', encoding='ISO-8859-1')
        print("File read successfully with ISO-8859-1 encoding.")
    except UnicodeDecodeError as e:
        print(f"Failed to read file with ISO-8859-1: {e}")

    # If the above fails, try with Windows-1252
    try:
        data = pd.read_csv('/home/oskar/NLP/datasets/Corona_NLP_train.csv', encoding='cp1252')
        print("File read successfully with Windows-1252 encoding.")
    except UnicodeDecodeError as e:
        print(f"Failed to read file with Windows-1252: {e}")

    data['Target'] = data['Sentiment'].replace({value:key for key, value in enumerate(data['Sentiment'].unique())})
    return data

def read_csv_text():
    file_path = "the-verdict.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as request:
            text_data = request.read().decode('utf-8')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text_data)
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            text_data = f.read()
    return text_data
