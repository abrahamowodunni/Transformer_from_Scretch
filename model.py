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
        return self.embedding(x) * math.sqrt(self.d_model)

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
        self.alpha = nn.Parameter(torch.ones(1)) # for multiplication
        self.bias = nn.Parameter(torch.zeros(1)) # for addition

    def forward(self,x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
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
        self.linear2 = nn.Linear(d_ff, d_model) # Linear transformation W2 with bias b2

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


    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch sq_len, d_model) --> (batch, sq_len, d_model)
        key = self.w_k(k) # (batch sq_len, d_model) --> (batch, sq_len, d_model)
        value = self.w_v(v) # (batch sq_len, d_model) --> (batch, sq_len, d_model)

        # (batch, sq_len, d_model) --> (batch, sq_len, h, d_k) --> (batch, h, sq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)


class ResidualConnection(nn.Module): # the skip connection!
    """
    Implements a residual connection with layer normalization and dropout.

    Args:
        dropout (float): Dropout probability to regularize the sublayer output.

    Methods:
        forward(x, sublayer):
            Applies layer normalization to the input 'x', passes it through the 'sublayer',
            applies dropout, and then adds the original input 'x' as a residual connection.

    Returns:
        Tensor: The output with a residual connection added to the transformed input.
    """
    def __init__(self, dropout):

        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    """
    Encoder block consisting of a self-attention mechanism and a feed-forward network with residual connections.

    Args:
        self_attention_block (MultiHeadAttentionBlock): Self-attention module that allows the input to attend to itself.
        feed_forward_block (FeedForwardBlock): Feed-forward network that transforms the input independently at each position.
        dropout (float): Dropout probability for regularizing the sublayers.

    Methods:
        forward(x, src_mask):
            Applies self-attention with a residual connection, followed by a feed-forward block with another residual connection.
    
    Returns:
        Tensor: The output after passing through the self-attention, feed-forward layers, and residual connections.
    """
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    """
    Implements the encoder stack of the Transformer model.

    Attributes:
        layers (nn.ModuleList): A list of stacked encoder blocks (self-attention + feed-forward + residual connections).
        norm (LayerNormalization): Layer normalization applied after the entire stack of encoder blocks.

    Methods:
        forward(x, mask): Passes the input through each encoder block, applying layer normalization to the final output.
    """
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    """
    Implements a single block of the Transformer decoder, consisting of self-attention, cross-attention, and feed-forward layers with residual connections.

    Attributes:
        self_attention_block (MultiHeadAttentionBlock): Self-attention mechanism for the target sequence.
        cross_attention_block (MultiHeadAttentionBlock): Attention mechanism between the target sequence and the encoder's output (cross-attention).
        feed_forward_block (FeedForwardBlock): Feed-forward layer for further transformations.
        residual_connections (nn.ModuleList): A list containing three residual connections for self-attention, cross-attention, and feed-forward blocks.

    Methods:
        forward(x, encoder_output, src_mask, tgt_mask): Passes the input through self-attention, cross-attention, and feed-forward blocks with residual connections.
    """
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block:MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.cross_attention_block = cross_attention_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    

    
class Decoder(nn.Module):
    """
    Decoder module consisting of multiple layers of decoder blocks, each containing self-attention, 
    cross-attention with the encoder output, and feed-forward layers. It includes residual connections 
    and normalization at each step.

    Args:
        layers (nn.Module): A list of DecoderBlock modules that will process the input sequentially.
    
    Methods:
        forward(x, encoder_output, src_mask, tgt_mask):
            Performs the forward pass through each decoder block, applying self-attention and cross-attention
            with the encoder output. Returns the normalized output after all layers are applied.

    Inputs:
        x (Tensor): Input sequence to the decoder (target sequence).
        encoder_output (Tensor): Output from the encoder (used in cross-attention).
        src_mask (Tensor): Mask for the source sequence (used in cross-attention to mask padding or unwanted tokens).
        tgt_mask (Tensor): Mask for the target sequence (used in self-attention to prevent attending to future tokens).

    Output:
        Tensor: The final output from the decoder after processing through all layers and normalization.
    """
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    """
    A projection layer that maps the hidden states (of dimension d_model) from the transformer decoder 
    to the vocabulary size, providing a probability distribution over the vocabulary for each position 
    in the sequence.

    Args:
        d_model (int): Dimensionality of the model's internal hidden states.
        vocab_size (int): Size of the output vocabulary.

    Methods:
        forward(x):
            Takes the hidden state output from the decoder (shape: batch, seq_len, d_model) 
            and projects it to the vocabulary size (batch, seq_len, vocab_size) using a linear transformation, 
            followed by log_softmax to get the probabilities for each token.

    Input:
        x (Tensor): The hidden state tensor with shape (batch, seq_len, d_model).

    Output:
        Tensor: Log-probabilities of tokens for each position in the sequence with shape 
        (batch, seq_len, vocab_size).
    """
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)
    
class Transformer(nn.Module):
    """
    A Transformer model consisting of an encoder and a decoder with input embeddings, 
    positional encodings, and a projection layer to produce vocabulary-level predictions.

    Args:
        encoder (Encoder): The encoder module of the transformer.
        decoder (Decoder): The decoder module of the transformer.
        src_embedding (InputEmbeddings): The embedding layer for the source input.
        tgt_embedding (InputEmbeddings): The embedding layer for the target input.
        src_position (PositionalEncoding): Positional encoding for the source input.
        tgt_position (PositionalEncoding): Positional encoding for the target input.
        projection_layer (ProjectionLayer): A linear layer that projects decoder outputs 
            to a probability distribution over the vocabulary.
    
    Methods:
        encode(src, src_mask):
            Encodes the source input sequence by embedding it, adding positional encodings, 
            and passing it through the encoder.

        decode(encoder_output, src_mask, tgt, tgt_mask):
            Decodes the target sequence by embedding it, adding positional encodings, 
            and passing it through the decoder with the encoder output.

        project(x):
            Projects the decoder output to a vocabulary-level probability distribution.
    
    Inputs:
        - `src`: Source sequence (batch, seq_len) for encoding.
        - `src_mask`: Mask for the source sequence.
        - `tgt`: Target sequence (batch, seq_len) for decoding.
        - `tgt_mask`: Mask for the target sequence.
    
    Outputs:
        - Log-probabilities of tokens for each position in the target sequence.
    """
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: InputEmbeddings, tgt_embedding: InputEmbeddings, src_position: PositionalEncoding, tgt_position: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed= src_embedding
        self.tgt_embed = tgt_embedding
        self.src_pos = src_position
        self.tgt_pos = tgt_position
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src,src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
def build_tranformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model = 512, N = 6, h = 8, dropout = 0.1, d_ff = 2048) -> Transformer:
    """
    Build a complete Transformer model for sequence-to-sequence tasks such as translation or text generation.

    Args:
        src_vocab_size (int): Size of the source vocabulary.
        tgt_vocab_size (int): Size of the target vocabulary.
        src_seq_len (int): Maximum length of the source sequences.
        tgt_seq_len (int): Maximum length of the target sequences.
        d_model (int, optional): Dimensionality of the embeddings and hidden layers. Default is 512.
        N (int, optional): Number of layers (blocks) in the encoder and decoder. Default is 6.
        h (int, optional): Number of attention heads in the multi-head attention mechanism. Default is 8.
        dropout (float, optional): Dropout probability. Default is 0.1.
        d_ff (int, optional): Dimensionality of the feed-forward network. Default is 2048.

    Returns:
        Transformer: A fully assembled Transformer model.

    """
    # embedding layer
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # positional encoding
    src_pos = PositionalEncoding(d_model, src_seq_len,dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # encoder block
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    # decoder block
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # create the projection LayerNormalization
    project_layer = ProjectionLayer(d_model,tgt_vocab_size)

    # creating the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, project_layer)

    # initialization of parmeters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)
    return transformer
