import torch
from torch import nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) 
        
        # x: (n, 1, 562), h0: (1, n, 128)
        
        # Forward propagate RNN
        # out, _ = self.rnn(x, h0)  
        # or:
        out, _ = self.lstm(x, (h0,c0))  
        
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n, 1, 128)
        
        # Decode the hidden state of the last time step

        out = out[:, -1, :]
         
        out = self.fc(out)
        return out

class TRANSFORMER(nn.Module):
    def __init__(self, d_model, n_head, num_classes, device):
        super(LSTM, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.device = device

        self.trans = nn.Transformer(d_model= self.d_model, nhead= self.n_head, batch_first=True)
        self.fc = nn.Linear(n_head, num_classes)
        
    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) 
        
        # x: (n, 1, 562), h0: (1, n, 128)
        
        # Forward propagate RNN
        # out, _ = self.rnn(x, h0)  
        # or:
        out, _ = self.trans(x, (h0,c0))  
        
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n, 1, 128)
        
        # Decode the hidden state of the last time step

        out = out[:, -1, :]
         
        out = self.fc(out)
        return out
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    modelLSTM = LSTM(input_size= 30 * 50, hidden_size= 128, num_layers= 4, num_classes= 2, device= device)
    modelLSTM = modelLSTM.to(device)
    # modelTrans = TRANSFORMER(d_model= 1500, n_head= 128, num_classes= 2, device= device).to(device)
    test1 = torch.rand((1,1,1500)).to(device)

    print(test1)
    modelLSTM.eval() #non-gradient

    print(modelLSTM(test1))

    transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
    src = torch.rand((10, 32, 512))
    tgt = torch.rand((20, 32, 512))
    out = transformer_model(src, tgt)
    print(out)