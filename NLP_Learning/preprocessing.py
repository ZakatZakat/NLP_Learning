from transformers import BertTokenizerFast
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

def read_csv(data=None, Target=None):
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

    