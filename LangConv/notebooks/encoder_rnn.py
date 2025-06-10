import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers=1, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(emb_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src)) 
        outputs, hidden = self.rnn(embedded)
        return hidden
