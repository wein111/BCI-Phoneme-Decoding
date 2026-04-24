# model_brainbert.py
# BrainBERT: Transformer-based neural signal decoder

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer.
    """
    def __init__(self, d_model, max_len=2000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (B, T, D)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class BrainBERT(nn.Module):
    """
    BrainBERT: Transformer Encoder for neural signal decoding.
    
    Architecture:
        Input Linear → Positional Encoding → Transformer Encoder → Output Linear
    
    Uses CTC loss for sequence-to-sequence training.
    """
    
    def __init__(
        self,
        input_dim=1280,        # 256 * 5 (after time stacking)
        d_model=256,           # Transformer hidden dimension
        nhead=8,               # Number of attention heads
        num_layers=6,          # Number of Transformer encoder layers
        dim_feedforward=1024,  # FFN hidden dimension
        dropout=0.1,
        num_classes=41         # 40 phonemes + 1 blank (CTC)
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights like BERT."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _create_padding_mask(self, lengths, max_len):
        """
        Create padding mask for Transformer.
        True = masked (padding), False = valid
        """
        batch_size = len(lengths)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        for i, length in enumerate(lengths):
            mask[i, length:] = True
        return mask
    
    def forward(self, x, lengths):
        """
        Args:
            x: (B, T, input_dim) - input features
            lengths: (B,) - actual sequence lengths
        
        Returns:
            logits: (T, B, num_classes) - for CTC loss
        """
        B, T, _ = x.shape
        
        # Input projection
        x = self.input_proj(x)  # (B, T, d_model)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Create padding mask
        padding_mask = self._create_padding_mask(lengths, T).to(x.device)
        
        # Transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # Output projection
        logits = self.output_proj(x)  # (B, T, num_classes)
        
        # CTC expects (T, B, C)
        return logits.permute(1, 0, 2)


class BrainBERTLite(nn.Module):
    """
    Lightweight version of BrainBERT for faster training.
    """
    
    def __init__(
        self,
        input_dim=1280,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        num_classes=41
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, num_classes)
    
    def _create_padding_mask(self, lengths, max_len):
        batch_size = len(lengths)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        for i, length in enumerate(lengths):
            mask[i, length:] = True
        return mask
    
    def forward(self, x, lengths):
        B, T, _ = x.shape
        
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        padding_mask = self._create_padding_mask(lengths, T).to(x.device)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        logits = self.output_proj(x)
        return logits.permute(1, 0, 2)

