import torch
import torch.nn as nn 
from torch.utils.data import Dataset

class BillingualDataSet(Dataset):
    """
    BillingualDataSet prepares bilingual text data for training a Transformer model for machine translation.

    Attributes:
        ds (Dataset): The parallel dataset containing source and target sentences.
        tokenizer_src (Tokenizer): Tokenizer for the source language.
        tokenizer_tgt (Tokenizer): Tokenizer for the target language.
        src_lang (str): The source language identifier.
        tgt_lang (str): The target language identifier.
        seq_len (int): The maximum length of token sequences.

    Methods:
        __len__: Returns the number of examples in the dataset.
        __getitem__: Returns a dictionary with encoder and decoder inputs, masks, and labels for a given index.
        causal_mask: Creates a causal mask for autoregressive decoding in the Transformer model.

    Returns:
        dict: A dictionary containing:
            - encoder_input: The padded and tokenized source sentence.
            - decoder_input: The tokenized target sentence (input to the decoder).
            - encoder_mask: The attention mask for the encoder (to ignore padding tokens).
            - decoder_mask: The causal mask for the decoder (to prevent attending to future tokens).
            - label: The tokenized target sentence used for the decoder's output.
            - src_text: The original source sentence.
            - tgt_text: The original target sentence.
    """
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype = torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype = torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype = torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        src_text = self.ds[idx]['src']
        tgt_text = self.ds[idx]['tgt']

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add sos, eos, and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 ## start and end of token here. 
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # only start of token

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Add <s> and </s> tokens
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
        ], dim=0)

        # Add only <s> token
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
        ], dim=0)

        # Add only </s> token what we expect at the end of the decoder
        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
        ], dim=0)

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return{
            "encoder_input": encoder_input, # seq_len
            "decoder_imput": decoder_input, # seq_len
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).int(), # we don't want to learn the padded ones
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & self.causal_mask(decoder_input.size(0)), # we just want it to focus on the previous works  
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text

        }
    
    def causal_mask(size):
        mask = torch.triu(torch.ones((1, size, size)), disgonal = 1).type(torch.int) 
        return mask == 0 # everything above the diagonal will become 0


