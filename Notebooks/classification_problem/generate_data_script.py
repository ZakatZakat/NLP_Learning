#!/usr/bin/env python
# coding: utf-8

# In[7]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd 
import os

from torch.utils.data import Dataset, TensorDataset, DataLoader

from transformers import BertTokenizer, BertForSequenceClassification, BertTokenizerFast

# Try with ISO-8859-1
data = pd.read_csv('./Notebooks/classification_problem/Corona_NLP_test.csv', encoding='ISO-8859-1')
#print("File read successfully with ISO-8859-1 encoding.")

data['Target'] = data['Sentiment'].replace({value:key for key, value in enumerate(data['Sentiment'].unique())})

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[9]:


def tokenize(text):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=520,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    return encoding

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

        embedded = tokenize(values)

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


# Parameters for the model
input_size = 520  # The number of features in the input (e.g., size of the embedding vector)
hidden_size = 100  # The number of features in the hidden state of the LSTM
output_size = 5  # The size of the output, e.g., 1 for a regression task

model = SimpleLSTMModel(input_size, hidden_size, output_size, 10).to(device)
dataset = TextClassificationDataset(data['OriginalTweet'].values, data['Target'].values, tokenizer)

train_dataloader = DataLoader(dataset, batch_size=10, shuffle=True, drop_last=True)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

gradient_norms = {}

# Функция для инициализации ключей в словаре
def init_gradient_norms(model):
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            gradient_norms[name] = []

# Функция для сохранения норм градиентов после каждого обратного распространения
def save_gradients(model):
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            gradient_norms[name].append(parameter.grad.norm().item())

init_gradient_norms(model)

for epoch in range(2):
    # Forward pass
    total = 0
    correct = 0
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(torch.float32)
        attention_mask = batch['masks']
        labels = batch['labels']

        input_ids, masks, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        outputs = model(input_ids).squeeze()
        loss = criterion(outputs, labels)
        
        # Backward pass и оптимизация
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        save_gradients(model)  # Печать нормы градиентов
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'epoch: {epoch}, accuracy: {accuracy}')

