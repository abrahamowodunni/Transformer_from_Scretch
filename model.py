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
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # so it doesn't change when training. 
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    """
    Applies Layer Normalization over the last dimension of the input tensor.

    Layer normalization stabilizes the learning process by normalizing the inputs to have zero mean and unit variance,
    followed by scaling and shifting with learnable parameters (alpha and bias).

    Args:
        eps (float): A small value added to the denominator for numerical stability (default: 1e-6).

    Attributes:
        alpha (nn.Parameter): Learnable scaling factor.
        bias (nn.Parameter): Learnable shift parameter.
    
    Forward Input:
        x (Tensor): Input tensor of shape (batch_size, ..., seq_len, d_model).

    Forward Output:
        Tensor: Normalized tensor of the same shape as the input.
    """
    def __init__(self, eps = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpah = nn.Parameter(torch.ones(1)) # for multiplication
        self.bias = nn.Parameter(torch.zeros(1)) # for addition

    def forward(self,x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.stf(dim = -1, keepdim = True)
        return self.alpha * (x-mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):
    """
    Implements a two-layer feedforward neural network with ReLU activation and dropout, typically used after the attention layer in a transformer.

    Args:
        d_model (int): The input and output dimensionality (model size).
        d_ff (int): The dimensionality of the feedforward layer (hidden size).
        dropout (float): The dropout rate applied after the ReLU activation.

    Forward Input:
        x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).

    Forward Output:
        Tensor: Transformed tensor of shape (batch_size, seq_len, d_model).
    
    Functionality:
        - First transforms input from (d_model) to (d_ff).
        - Applies ReLU for non-linearity.
        - Applies dropout for regularization.
        - Finally transforms back to (d_model).
    """
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # Linear transformation W1 with bias b1
        self.dropout = nn.Dropout(dropout) # Dropout for regularization
        self.linear2 = nn.Linear(d_model, d_ff) # Linear transformation W2 with bias b2

    def forward(self, x):
        # (batch, sq_len, d_model) --> (batch, sq_len, d_ff) --> (batch, sq_len, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
    

class MultiHeadAttentionBlock(nn.Module):
    """
    Implements the Multi-Head Attention mechanism used in transformers.
    
    Args:
        d_model (int): The dimensionality of the input and output (model size).
        h (int): The number of attention heads.
        dropout (float): The dropout rate applied during attention computation.

    Forward Input:
        q (Tensor): Query tensor of shape (batch_size, seq_len, d_model).
        k (Tensor): Key tensor of shape (batch_size, seq_len, d_model).
        v (Tensor): Value tensor of shape (batch_size, seq_len, d_model).
        mask (Tensor, optional): Mask tensor to avoid attending to padding positions.

    Forward Output:
        Tensor: Output tensor of shape (batch_size, seq_len, d_model) after attention.
    
    Functionality:
        - Splits input into multiple heads (h).
        - Computes scaled dot-product attention for each head.
        - Concatenates the heads' outputs and applies a final linear projection.
    """
    def __init__(self, d_model, h, dropout):
        super().__init__()
        self.d_model =d_model
        self.h = h
        assert d_model % h == 0, "d_modle is not divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1] # d_k = dimension of each head (derived from d_model)

        # Compute the dot product between query and key, scaled by √d_k
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)

        # Apply the mask (if provided) to avoid attending to certain positions
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1)

        # Apply dropout to the attention scores for regularization
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores


    def foward(self, q, k, v, mask):
        query = self.w_q(q) # (batch sq_len, d_model) --> (batch, sq_len, d_model)
        key = self.w_k(k) # (batch sq_len, d_model) --> (batch, sq_len, d_model)
        value = self.w_v(v) # (batch sq_len, d_model) --> (batch, sq_len, d_model)

        # (batch, sq_len, d_model) --> (batch, sq_len, h, d_k) --> (batch, h, sq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1,2).contiguous().view(x.ahape[0], -1, self.h * self.d_k)

        return self.w_o(x)