from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

@dataclass
class RTDetrV2Config:
    """
    Complete standalone configuration class for RT-DETR v2.
    """
    # General / Output
    num_labels: int = 68
    hidden_dim: int = 256
    d_model: int = 256
    num_queries: int = 300
    
    # Layer Norm & Batch Norm
    layer_norm_eps: float = 1e-5
    batch_norm_eps: float = 1e-5
    normalize_before: bool = False
    
    # Encoder Parameters
    encoder_hidden_dim: int = 256
    encoder_in_channels: Tuple[int, ...] = (128, 256, 512)
    feat_strides: Tuple[int, ...] = (8, 16, 32)
    encoder_layers: int = 1
    encoder_ffn_dim: int = 1024
    num_attention_heads: int = 8
    encoder_attention_heads: int = 8
    encode_proj_layers: Tuple[int, ...] = (2,)
    positional_encoding_temperature: int = 10000
    encoder_activation_function: str = "relu"
    activation_function: str = "silu"
    hidden_expansion: float = 0.5
    eval_size: Optional[int] = None
    
    # Decoder Parameters
    decoder_in_channels: Tuple[int, ...] = (256, 256, 256)
    decoder_layers: int = 6
    decoder_ffn_dim: int = 1024
    decoder_attention_heads: int = 8
    decoder_activation_function: str = "relu"
    decoder_n_points: int = 4
    decoder_n_levels: int = 3
    decoder_offset_scale: float = 0.5
    decoder_method: str = "discrete"
    
    # Dropout
    dropout: float = 0.0
    activation_dropout: float = 0.0
    attention_dropout: float = 0.0
    
    # Label Mappings
    id2label: Dict[int, str] = field(default_factory=dict)
    label2id: Dict[str, int] = field(default_factory=dict)