import torch
import torch.nn as nn
import torch.optim as optim

class SleepGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(SleepGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])  # Only take the output from the last time step
        return out

# Hyperparameters
input_size = X_train.shape[1]  # Number of features
hidden_size = 64
output_size = 1  # Assuming regression for sleep quality
num_layers = 2
model = SleepGRU(input_size, hidden_size, output_size, num_layers).to('cpu')
