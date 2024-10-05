import torch
import torch.nn as nn 
import math

class InputEmbeddings(nn.Module):
    """
    A PyTorch module for generating input embeddings for tokens in a sequence.

    Args:
        d_model (int): Dimension of the embedding vectors.
        vocab_size (int): The size of the vocabulary.

    Forward:
        x (Tensor): Input tensor of token indices.
    
    Returns:
        Tensor: Scaled embeddings for the input tokens.
    """
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model) # torch can help us with this 

    def forward(self, x):
        return self.embedding * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """
    A PyTorch module for adding positional encodings to token embeddings.

    Args:
        d_model (int): Dimension of the embedding vectors.
        seq_len (int): Maximum length of the input sequence.
        dropout (float): Dropout rate for regularization.

    Forward:
        x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).

    Returns:
        Tensor: Token embeddings with added positional encodings and dropout applied.
    """
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # creating a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len,d_model)
        # create a vector of shape (seq_le, 1n)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        denominator = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        # apply the sin to even and cos to odd
        pe[:,0::2] = torch.sin(position * denominator)
        pe[:,1::2] = torch.cos(position * denominator)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('pe',pe) # so it is not treated as a prameter, meaning no update durring backpropagation. 

    def forward(self,x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

