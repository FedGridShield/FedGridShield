import torch.nn as nn

class ETD_Model(nn.Module):
    def __init__(self, input_features=1, dense_features=10, lstm_units=300, num_classes=2):
        super(ETD_Model, self).__init__()
        self.dense1 = nn.Linear(input_features, dense_features)
        self.tanh = nn.Tanh()
        self.lstm1 = nn.LSTM(dense_features, lstm_units, batch_first=True)
        self.lstm2 = nn.LSTM(lstm_units, lstm_units, batch_first=True)
        self.dense2 = nn.Linear(lstm_units, num_classes)

    def forward(self, x):
        x = self.dense1(x)  # Output: (batch_size, 48, dense_features)
        x = self.tanh(x)
        x, _ = self.lstm1(x) # Output: (batch_size, 48, lstm_units)
        output_seq, (h_n, c_n) = self.lstm2(x)
        x = output_seq[:, -1, :] # Output: (batch_size, lstm_units)
        x = self.dense2(x) # Output: (batch_size, num_classes)
        return x
