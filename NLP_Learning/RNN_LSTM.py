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