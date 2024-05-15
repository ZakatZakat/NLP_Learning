from transformers import BertTokenizerFast
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

def tokenizer(text, max_length=520):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    return encoding

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


# Define dataset class
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        values = self.texts[idx]
        label = self.labels[idx]

        embedded = tokenizer(values)

        input_ids = embedded['input_ids'].flatten()
        masks = embedded['attention_mask'].flatten()

        return {'input_ids': input_ids, 'masks': masks, 'labels': torch.tensor(label, dtype=torch.long)}


class SimpleLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(SimpleLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Forward propagate LSTM
        out, (hn, cn) = self.lstm(x)

        # Pass through the fully connected layer
        out = self.fc(hn)
        
        # Apply the ReLU activation function
        out = self.relu(out)
        
        return out