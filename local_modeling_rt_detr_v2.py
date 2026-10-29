import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import warnings
import math
import functools
from collections.abc import Callable
from dataclasses import dataclass
from backbone import RTDetrMobileNetBackbone
from local_config_rt_detr_v2 import RTDetrV2Config




class RTDetrV2ForObjectDetection(nn.Module):
    """
    Pure PyTorch implementation of RT-DETR v2.
    """
    def __init__(self, config: RTDetrV2Config):
        super().__init__()
        self.config = config
        
        # 1. Initialize your backbone (Build in Pipeline 3)
        self.backbone = RTDetrMobileNetBackbone(pretrained=False)

        self.encoder_input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, config.encoder_hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(config.encoder_hidden_dim)
            ) for in_ch in [24, 48, 96]  # MobileNet output channels
        ])
        
        # 2. Initialize your encoder (aifi)
        self.encoder = RTDetrV2HybridEncoder(config)

        # ---> TWO-STAGE QUERY SELECTION BRIDGE <---
        
        # 1. Project the encoder's output channels for the decoder
        self.decoder_input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, config.d_model, kernel_size=1, bias=False),
                nn.BatchNorm2d(config.d_model, config.batch_norm_eps)
            ) for in_channels in config.decoder_in_channels
        ])
        
        # 2. The memory processing bottleneck
        self.enc_output = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model, eps=config.layer_norm_eps),
        )
        
        # 3. The preliminary classification head (Foreground vs Background)
        self.enc_score_head = nn.Linear(config.d_model, config.num_labels)
        
        # 4. The preliminary bounding box head
        self.enc_bbox_head = RTDetrV2MLPPredictionHead(
            input_dim=config.d_model, 
            hidden_dim=config.d_model, 
            output_dim=4, 
            num_layers=3
        )
        # 3. Initialize your decoder
        self.decoder = RTDetrV2Decoder(config)

        self.class_embed = nn.ModuleList([
            nn.Linear(config.hidden_dim, config.num_labels) 
            for _ in range(config.decoder_layers)
        ])
        
        self.bbox_embed = nn.ModuleList([
            RTDetrV2MLPPredictionHead(
                input_dim=config.hidden_dim, 
                hidden_dim=config.hidden_dim, 
                output_dim=4, 
                num_layers=3
            ) for _ in range(config.decoder_layers)
        ])

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for layer in self.class_embed:
            nn.init.constant_(layer.bias, bias_value)

    def generate_anchors(self, spatial_shapes, grid_size=0.05, device="cpu"):
        """
        Generates the initial grid coordinates (anchors) for the encoder's preliminary bounding box predictions.
        """
        anchors = []
        for level, (height, width) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(end=height, device=device, dtype=torch.float32),
                torch.arange(end=width, device=device, dtype=torch.float32),
                indexing="ij",
            )
            grid_xy = torch.stack([grid_x, grid_y], -1)
            grid_xy = grid_xy.unsqueeze(0) + 0.5
            grid_xy[..., 0] /= width
            grid_xy[..., 1] /= height
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**level)
            anchors.append(torch.cat([grid_xy, wh], -1).reshape(-1, height * width, 4))
            
        anchors = torch.cat(anchors, 1)
        # Skip the masking logic entirely to prevent SELECT_V2
        anchors = torch.log(anchors / (1 - anchors))
        # Return a dummy mask to prevent SELECT_V2
        return anchors, torch.ones_like(anchors[..., :1], dtype=torch.bool)

    def forward(self, pixel_values: torch.Tensor):
        """
        A clean forward pass handling backbone extraction, two-stage query selection, and decoding.
        """
        device = pixel_values.device
        
        # 1. Extract and project backbone features
        raw_feature_maps = self.backbone(pixel_values)
        projected_feature_maps = [
            proj(feat) for proj, feat in zip(self.encoder_input_proj, raw_feature_maps)
        ]

        # 2. Pass through the Hybrid Encoder
        encoded_features = self.encoder(projected_feature_maps)
        
        # 3. Two-Stage Bridge: Prepare decoder inputs
        sources = [self.decoder_input_proj[level](source) for level, source in enumerate(encoded_features)]
        
        source_flatten = []
        spatial_shapes_list = []
        spatial_shapes = torch.empty((len(sources), 2), device=device, dtype=torch.long)
        
        for level, source in enumerate(sources):
            height, width = source.shape[-2:]
            spatial_shapes[level, 0] = height
            spatial_shapes[level, 1] = width
            spatial_shapes_list.append((height, width))
            source_flatten.append(source.flatten(2).transpose(1, 2))
            
        source_flatten = torch.cat(source_flatten, 1)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        # 4. Generate anchors and process memory bottleneck
        anchors, valid_mask = self.generate_anchors(spatial_shapes_list, device=device)
        memory = valid_mask.to(source_flatten.dtype) * source_flatten
        output_memory = self.enc_output(memory)

        # 5. Preliminary Predictions (Foreground vs Background)
        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_logits = self.enc_bbox_head(output_memory) + anchors

        # 6. Extract the top queries to feed the decoder
        # Find out exactly how many tokens the CNN backbone produced
        available_queries = enc_outputs_class.shape[1]
        
        # Cap the query selection so we never index out of bounds
        safe_num_queries = min(self.config.num_queries, available_queries)
        
        # Generate the static index up to the safe limit
        topk_ind = torch.arange(safe_num_queries, device=device).unsqueeze(0)
        
        batch_size = enc_outputs_class.shape[0]
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)
        
        # Keep these extractions intact!
        reference_points_unact = enc_outputs_coord_logits[batch_idx, topk_ind]
        init_reference_points = torch.sigmoid(reference_points_unact).detach()

        target = output_memory[batch_idx, topk_ind].detach()
        """
        # 6. Extract the top queries to feed the decoder
        #    DIAGNOSTIC BYPASS: Comment out the dynamic TopK sorting
        # _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.config.num_queries, dim=1)
    
        # Replace it with a static index of the first N queries
        topk_ind = torch.arange(self.config.num_queries, device=device).unsqueeze(0)
    
        batch_size = enc_outputs_class.shape[0]
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)
    
        reference_points_unact = enc_outputs_coord_logits[batch_idx, topk_ind]
        init_reference_points = torch.sigmoid(reference_points_unact).detach()

        target = output_memory[batch_idx, topk_ind].detach()
        """
        # 7. Final Decoder Pass (Iterative Refinement)
        inter_hidden, inter_refs = self.decoder(
            hidden_states=target,
            encoder_hidden_states=source_flatten,
            reference_points=init_reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            bbox_embed=self.bbox_embed 
        )

        # Apply the unique classification head to each layer's hidden state
        all_logits = torch.stack([
            self.class_embed[idx](state) for idx, state in enumerate(inter_hidden)
        ])

    # Package final outputs, intermediate auxiliary outputs, and encoder proposals
        return {
            "logits": all_logits[-1],
            "pred_boxes": inter_refs[-1],
            "aux_outputs": [
                {"logits": all_logits[i], "pred_boxes": inter_refs[i]} 
                for i in range(len(all_logits) - 1)
            ],
            # ADD THIS NEW BLOCK:
            "enc_outputs": {
                "logits": enc_outputs_class,
                # Apply sigmoid here so it matches the 0-1 scale of the decoder boxes
                "pred_boxes": torch.sigmoid(enc_outputs_coord_logits) 
            }
        }
class RTDetrV2EncoderLayer(nn.Module):
    def __init__(self, config: RTDetrV2Config):
        super().__init__()
        self.normalize_before = config.normalize_before
        self.hidden_size = config.encoder_hidden_dim

        # self-attention
        self.self_attn = RTDetrV2SelfAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_attention_heads=config.num_attention_heads,
            dropout=config.dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.dropout = config.dropout
        self.mlp = RTDetrV2MLP(config, self.hidden_size, config.encoder_ffn_dim, config.encoder_activation_function)
        self.final_layer_norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        spatial_position_embeddings: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, hidden_size)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative
                values.
            spatial_position_embeddings (`torch.FloatTensor`, *optional*):
                Spatial position embeddings (2D positional encodings of image locations), to be added to both
                the queries and keys in self-attention (but not to values).
        """
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=spatial_position_embeddings,
            **kwargs,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        if not self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        if self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)
        residual = hidden_states

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states
        if not self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.training:
            if not torch.isfinite(hidden_states).all():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return hidden_states

class RTDetrV2DecoderLayer(nn.Module):
    def __init__(self, config: RTDetrV2Config):
        super().__init__()
        self.hidden_size = config.d_model

        # self-attention
        self.self_attn = RTDetrV2SelfAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_attention_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        # override only the encoder attention module with v2 version
        self.encoder_attn = RTDetrV2MultiscaleDeformableAttention(config)
        self.encoder_attn_layer_norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        # feedforward neural networks
        self.mlp = RTDetrV2MLP(config, self.hidden_size, config.decoder_ffn_dim, config.decoder_activation_function)
        self.final_layer_norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        object_queries_position_embeddings: torch.Tensor | None = None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=encoder_attention_mask,
            position_embeddings=object_queries_position_embeddings,
            **kwargs,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states

        # Cross-Attention
        hidden_states, _ = self.encoder_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            position_embeddings=object_queries_position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            **kwargs,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states
class RTDetrV2MLPPredictionHead(nn.Module):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.

    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x




class RTDetrV2SelfAttention(nn.Module):
    """
    Multi-headed self-attention from 'Attention Is All You Need' paper.

    In RT_DETR_V2, position embeddings are added to both queries and keys (but not values) in self-attention.
    """

    def __init__(
        self,
        config: RTDetrV2Config,
        hidden_size: int,
        num_attention_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.config = config
        self.head_dim = hidden_size // num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = dropout
        self.is_causal = False

        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Position embeddings are added to both queries and keys (but not values).
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_key_input = hidden_states + position_embeddings if position_embeddings is not None else hidden_states

        query_states = self.q_proj(query_key_input).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(query_key_input).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class RTDetrV2MLP(nn.Module):
    def __init__(self, config: RTDetrV2Config, hidden_size: int, intermediate_size: int, activation_function: str):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.activation_fn = getattr(F, activation_function, F.relu)
        self.activation_dropout = config.activation_dropout
        self.dropout = config.dropout

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        return hidden_states


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs,
):
    if scaling is None:
        scaling = query.size(-1) ** -0.5

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

class RTDetrV2MultiscaleDeformableAttention(nn.Module):
    """
    RTDetrV2 version of multiscale deformable attention, extending the base implementation
    with improved offset handling and initialization.
    """

    def __init__(self, config: RTDetrV2Config):
        super().__init__()
        num_heads = config.decoder_attention_heads
        n_points = config.decoder_n_points

        if config.d_model % num_heads != 0:
            raise ValueError(
                f"embed_dim (d_model) must be divisible by num_heads, but got {config.d_model} and {num_heads}"
            )
        dim_per_head = config.d_model // num_heads
        # check if dim_per_head is power of 2
        if not ((dim_per_head & (dim_per_head - 1) == 0) and dim_per_head != 0):
            warnings.warn(
                "You'd better set embed_dim (d_model) in RTDetrV2MultiscaleDeformableAttention to make the"
                " dimension of each attention head a power of 2 which is more efficient in the authors' CUDA"
                " implementation."
            )

        self.im2col_step = 64

        self.d_model = config.d_model

        # V2-specific attributes
        self.n_levels = config.decoder_n_levels
        self.n_heads = num_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(config.d_model, num_heads * self.n_levels * n_points * 2)
        self.attention_weights = nn.Linear(config.d_model, num_heads * self.n_levels * n_points)
        self.value_proj = nn.Linear(config.d_model, config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.d_model)

        self.offset_scale = config.decoder_offset_scale
        self.method = config.decoder_method

        # Initialize n_points list and scale
        n_points_list = [self.n_points for _ in range(self.n_levels)]
        self.n_points_list = n_points_list
        n_points_scale = [1 / n for n in n_points_list for _ in range(n)]
        self.register_buffer("n_points_scale", torch.tensor(n_points_scale, dtype=torch.float32))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings: torch.Tensor | None = None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        **kwargs,
    ):
        # Process inputs up to sampling locations calculation using parent class logic
        if position_embeddings is not None:
            hidden_states = hidden_states + position_embeddings

        batch_size, num_queries, _ = hidden_states.shape
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == sequence_length, "Make sure to align the spatial shapes with the sequence length"

        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            value = value.masked_fill(~attention_mask[..., None], float(0))
        value = value.view(batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads)

        # V2-specific sampling offsets shape
        sampling_offsets = self.sampling_offsets(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points, 2
        )

        attention_weights = self.attention_weights(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1)

        # V2-specific sampling locations calculation
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            n_points_scale = self.n_points_scale.to(dtype=hidden_states.dtype).unsqueeze(-1)
            offset = sampling_offsets * n_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")

        # V2-specific attention implementation choice
        output = multi_scale_deformable_attention_v2(
            value, spatial_shapes_list, sampling_locations, attention_weights, self.n_points_list, self.method
        )

        output = self.output_proj(output)
        return output, attention_weights

def multi_scale_deformable_attention_v2(
    value: Tensor,
    value_spatial_shapes: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
    num_points_list: list[int],
    method="default",
) -> Tensor:
    batch_size, _, num_heads, hidden_dim = value.shape
    _, num_queries, num_heads, num_levels, num_points = sampling_locations.shape
    value_list = (
        value.permute(0, 2, 3, 1)
        .flatten(0, 1)
        .split([height * width for height, width in value_spatial_shapes], dim=-1)
    )
    # sampling_offsets [8, 480, 8, 12, 2]
    if method == "default":
        sampling_grids = 2 * sampling_locations - 1
    elif method == "discrete":
        sampling_grids = sampling_locations
    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_grids = sampling_grids.split(num_points_list, dim=-2)
    sampling_value_list = []
    for level_id, (height, width) in enumerate(value_spatial_shapes):
        # batch_size, height*width, num_heads, hidden_dim
        # -> batch_size, height*width, num_heads*hidden_dim
        # -> batch_size, num_heads*hidden_dim, height*width
        # -> batch_size*num_heads, hidden_dim, height, width
        value_l_ = value_list[level_id].reshape(batch_size * num_heads, hidden_dim, height, width)
        # batch_size, num_queries, num_heads, num_points, 2
        # -> batch_size, num_heads, num_queries, num_points, 2
        # -> batch_size*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[level_id]
        # batch_size*num_heads, hidden_dim, num_queries, num_points
        if method == "default":
            sampling_value_l_ = nn.functional.grid_sample(
                value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
            )
        elif method == "discrete":
            sampling_coord = (sampling_grid_l_ * torch.tensor([[width, height]], device=value.device) + 0.5).to(
                torch.int64
            )

            # Separate clamping for x and y coordinates
            sampling_coord_x = sampling_coord[..., 0].clamp(0, width - 1)
            sampling_coord_y = sampling_coord[..., 1].clamp(0, height - 1)

            # Combine the clamped coordinates
            sampling_coord = torch.stack([sampling_coord_x, sampling_coord_y], dim=-1)
            sampling_coord = sampling_coord.reshape(batch_size * num_heads, num_queries * num_points_list[level_id], 2)
            sampling_idx = (
                torch.arange(sampling_coord.shape[0], device=value.device)
                .unsqueeze(-1)
                .repeat(1, sampling_coord.shape[1])
            )
            sampling_value_l_ = value_l_[sampling_idx, :, sampling_coord[..., 1], sampling_coord[..., 0]]
            sampling_value_l_ = sampling_value_l_.transpose(1, 2).reshape(
                batch_size * num_heads, hidden_dim, num_queries, num_points_list[level_id]
            )
        sampling_value_list.append(sampling_value_l_)
    # (batch_size, num_queries, num_heads, num_levels, num_points)
    # -> (batch_size, num_heads, num_queries, num_levels, num_points)
    # -> (batch_size, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.permute(0, 2, 1, 3).reshape(
        batch_size * num_heads, 1, num_queries, sum(num_points_list)
    )
    output = (
        (torch.concat(sampling_value_list, dim=-1) * attention_weights)
        .sum(-1)
        .view(batch_size, num_heads * hidden_dim, num_queries)
    )
    return output.transpose(1, 2).contiguous()



class RTDetrV2ConvNormLayer(nn.Module):
    def __init__(self, config, in_channels, out_channels, kernel_size, stride, padding=None, activation=None):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels, config.batch_norm_eps)
        self.activation = nn.Identity() if activation is None else getattr(F, activation, F.relu)

    def forward(self, hidden_state):
        hidden_state = self.conv(hidden_state)
        hidden_state = self.norm(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state

class RTDetrV2RepVggBlock(nn.Module):
    """
    RepVGG architecture block introduced by the work "RepVGG: Making VGG-style ConvNets Great Again".
    """

    def __init__(self, config: RTDetrV2Config):
        super().__init__()

        activation = config.activation_function
        hidden_channels = int(config.encoder_hidden_dim * config.hidden_expansion)
        self.conv1 = RTDetrV2ConvNormLayer(config, hidden_channels, hidden_channels, 3, 1, padding=1)
        self.conv2 = RTDetrV2ConvNormLayer(config, hidden_channels, hidden_channels, 1, 1, padding=0)
        self.activation = nn.Identity() if activation is None else getattr(F, activation, F.relu)

    def forward(self, x):
        y = self.conv1(x) + self.conv2(x)
        return self.activation(y)

class RTDetrV2CSPRepLayer(nn.Module):
    """
    Cross Stage Partial (CSP) network layer with RepVGG blocks.
    """

    def __init__(self, config: RTDetrV2Config):
        super().__init__()

        in_channels = config.encoder_hidden_dim * 2
        out_channels = config.encoder_hidden_dim
        num_blocks = 3
        activation = config.activation_function

        hidden_channels = int(out_channels * config.hidden_expansion)
        self.conv1 = RTDetrV2ConvNormLayer(config, in_channels, hidden_channels, 1, 1, activation=activation)
        self.conv2 = RTDetrV2ConvNormLayer(config, in_channels, hidden_channels, 1, 1, activation=activation)
        self.bottlenecks = nn.Sequential(*[RTDetrV2RepVggBlock(config) for _ in range(num_blocks)])
        if hidden_channels != out_channels:
            self.conv3 = RTDetrV2ConvNormLayer(config, hidden_channels, out_channels, 1, 1, activation=activation)
        else:
            self.conv3 = nn.Identity()

    def forward(self, hidden_state):
        hidden_state_1 = self.conv1(hidden_state)
        hidden_state_1 = self.bottlenecks(hidden_state_1)
        hidden_state_2 = self.conv2(hidden_state)
        return self.conv3(hidden_state_1 + hidden_state_2)


def build_2d_sinusoidal_position_embedding(
    height: int,
    width: int,
    embed_dim: int = 256,
    temperature: float = 10000.0,
    cls_token: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """2D sinusoidal position embeddings for an image patch grid.

    Each (h, w) position gets an ``embed_dim``-dimensional vector laid out as
    ``[sin_h | cos_h | sin_w | cos_w]``, with row-major (H-outer) patch ordering.

    Args:
        height: Grid height in patches.
        width: Grid width in patches.
        embed_dim: Total embedding dimension; must be divisible by 4.
        temperature: Base for the frequency decay.
        cls_token: If `True`, prepend a zero row for a CLS token.
        device: Target device; defaults to CPU.
        dtype: Output dtype; frequency arithmetic uses float64 internally.

    Returns:
        Tensor of shape ``(height * width [+1], embed_dim)``.
    """
    if embed_dim % 4 != 0:
        raise ValueError(f"`embed_dim` must be divisible by 4, got {embed_dim}")

    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float64, device=device) / pos_dim
    omega = 1.0 / temperature**omega  # (D/4,)

    grid_h = torch.arange(height, dtype=torch.float64, device=device)
    grid_w = torch.arange(width, dtype=torch.float64, device=device)
    grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing="ij")  # (H, W) each

    emb_h = grid_h.flatten().outer(omega)  # (H*W, D/4)
    emb_w = grid_w.flatten().outer(omega)  # (H*W, D/4)

    pos_embed = torch.cat([emb_h.sin(), emb_h.cos(), emb_w.sin(), emb_w.cos()], dim=1)

    if cls_token:
        pos_embed = torch.cat([torch.zeros(1, embed_dim, dtype=torch.float64, device=device), pos_embed], dim=0)

    return pos_embed.to(dtype)

class RTDetrV2SinePositionEmbedding(nn.Module):
    """
    2D sinusoidal position embedding used in RT-DETR hybrid encoder.
    """

    def __init__(self, embed_dim: int = 256, temperature: int = 10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature

    @staticmethod
    #problem below is that the lru_cache decorator is not compatible with torch.jit.script, so we need to use a custom decorator that is compatible with torch.jit.script
    @functools.lru_cache(maxsize=32)
    def _cached_build_2d_sinusoidal_position_embedding(*args, **kwargs) -> torch.Tensor:
        return build_2d_sinusoidal_position_embedding(*args, **kwargs)

    def forward(
        self,
        width: int,
        height: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Generate 2D sinusoidal position embeddings.

        Returns:
            Position embeddings of shape (1, height*width, embed_dim)
        """


        return self._cached_build_2d_sinusoidal_position_embedding(
            height=int(height),
            #torch_int is a custom function that converts a float to an int, and raises an error if the float is not an integer
            width=int(width),
            embed_dim=self.embed_dim,
            temperature=self.temperature,
            device=device,
            dtype=dtype,
        ).unsqueeze(0)


class RTDetrV2AIFILayer(nn.Module):
    """
    AIFI (Attention-based Intra-scale Feature Interaction) layer used in RT-DETR hybrid encoder.
    """

    def __init__(self, config: RTDetrV2Config):
        super().__init__()
        self.config = config
        self.encoder_hidden_dim = config.encoder_hidden_dim
        self.eval_size = config.eval_size

        self.position_embedding = RTDetrV2SinePositionEmbedding(
            embed_dim=self.encoder_hidden_dim,
            temperature=config.positional_encoding_temperature,
        )
        self.layers = nn.ModuleList([RTDetrV2EncoderLayer(config) for _ in range(config.encoder_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, channels, height, width)`):
                Feature map to process.
        """
        batch_size = hidden_states.shape[0]
        height, width = hidden_states.shape[2:]

        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        if self.training or self.eval_size is None:
            pos_embed = self.position_embedding(
                width=width,
                height=height,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        else:
            pos_embed = None

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=None,
                spatial_position_embeddings=pos_embed,
                **kwargs,
            )

        hidden_states = (
            hidden_states.transpose(1, 2).reshape(batch_size, self.encoder_hidden_dim, height, width).contiguous()
        )

        return hidden_states

class RTDetrV2HybridEncoder(nn.Module):
    """
    Hybrid encoder consisting of AIFI (Attention-based Intra-scale Feature Interaction) layers,
    a top-down Feature Pyramid Network (FPN) and a bottom-up Path Aggregation Network (PAN).
    More details on the paper: https://huggingface.co/papers/2304.08069

    Args:
        config: RTDetrV2Config
    """

    _can_record_outputs = {
        "hidden_states": RTDetrV2AIFILayer,
        "attentions": RTDetrV2SelfAttention,
    }

    def __init__(self, config: RTDetrV2Config):
        super().__init__()
        self.config = config
        self.in_channels = config.encoder_in_channels
        self.feat_strides = config.feat_strides
        self.encoder_hidden_dim = config.encoder_hidden_dim
        self.encode_proj_layers = config.encode_proj_layers
        self.positional_encoding_temperature = config.positional_encoding_temperature
        self.eval_size = config.eval_size
        self.out_channels = [self.encoder_hidden_dim for _ in self.in_channels]
        self.out_strides = self.feat_strides
        self.num_fpn_stages = len(self.in_channels) - 1
        self.num_pan_stages = len(self.in_channels) - 1

        # AIFI (Attention-based Intra-scale Feature Interaction) layers
        self.aifi = nn.ModuleList([RTDetrV2AIFILayer(config) for _ in range(len(self.encode_proj_layers))])

        # top-down FPN
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(self.num_fpn_stages):
            lateral_conv = RTDetrV2ConvNormLayer(
                config,
                in_channels=self.encoder_hidden_dim,
                out_channels=self.encoder_hidden_dim,
                kernel_size=1,
                stride=1,
                activation=config.activation_function,
            )
            fpn_block = RTDetrV2CSPRepLayer(config)
            self.lateral_convs.append(lateral_conv)
            self.fpn_blocks.append(fpn_block)

        # bottom-up PAN
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(self.num_pan_stages):
            downsample_conv = RTDetrV2ConvNormLayer(
                config,
                in_channels=self.encoder_hidden_dim,
                out_channels=self.encoder_hidden_dim,
                kernel_size=3,
                stride=2,
                activation=config.activation_function,
            )
            pan_block = RTDetrV2CSPRepLayer(config)
            self.downsample_convs.append(downsample_conv)
            self.pan_blocks.append(pan_block)

        

    
    def forward(
        self,
        inputs_embeds=None,
        **kwargs,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Flattened feature map (output of the backbone + projection layer) that is passed to the encoder.
        """
        feature_maps = inputs_embeds

        # AIFI: Apply transformer encoder to specified feature levels
        if self.config.encoder_layers > 0:
            for i, enc_ind in enumerate(self.encode_proj_layers):
                feature_maps[enc_ind] = self.aifi[i](feature_maps[enc_ind], **kwargs)

        # top-down FPN
        fpn_feature_maps = [feature_maps[-1]]
        for idx, (lateral_conv, fpn_block) in enumerate(zip(self.lateral_convs, self.fpn_blocks)):
            backbone_feature_map = feature_maps[self.num_fpn_stages - idx - 1]
            top_fpn_feature_map = fpn_feature_maps[-1]
            # apply lateral block
            top_fpn_feature_map = lateral_conv(top_fpn_feature_map)
            fpn_feature_maps[-1] = top_fpn_feature_map
            # apply fpn block
            top_fpn_feature_map = F.interpolate(top_fpn_feature_map, scale_factor=2.0, mode="nearest")
            fused_feature_map = torch.concat([top_fpn_feature_map, backbone_feature_map], dim=1)
            new_fpn_feature_map = fpn_block(fused_feature_map)
            fpn_feature_maps.append(new_fpn_feature_map)

        fpn_feature_maps.reverse()

        # bottom-up PAN
        pan_feature_maps = [fpn_feature_maps[0]]
        for idx, (downsample_conv, pan_block) in enumerate(zip(self.downsample_convs, self.pan_blocks)):
            top_pan_feature_map = pan_feature_maps[-1]
            fpn_feature_map = fpn_feature_maps[idx + 1]
            downsampled_feature_map = downsample_conv(top_pan_feature_map)
            fused_feature_map = torch.concat([downsampled_feature_map, fpn_feature_map], dim=1)
            new_pan_feature_map = pan_block(fused_feature_map)
            pan_feature_maps.append(new_pan_feature_map)

        return pan_feature_maps



def inverse_sigmoid(x, eps=1e-5):
    # We completely removed the [0, 1] forcing logic here
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

class RTDetrV2Decoder(nn.Module):
    def __init__(self, config: RTDetrV2Config):
        super().__init__()
        self.layers = nn.ModuleList([RTDetrV2DecoderLayer(config) for _ in range(config.decoder_layers)])
        self.query_pos_head = RTDetrV2MLPPredictionHead(4, 2 * config.d_model, config.d_model, num_layers=2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        spatial_shapes_list: list,
        level_start_index: torch.Tensor,
        bbox_embed: nn.ModuleList 
    ):
        intermediate_hidden_states = []
        intermediate_ref_points = []
        
        for idx, layer in enumerate(self.layers):
            obj_pos_embed = self.query_pos_head(reference_points)
            ref_points_input = reference_points.unsqueeze(2)
            
            # Standard decoder layer forward pass
            hidden_states = layer(
                hidden_states,
                object_queries_position_embeddings=obj_pos_embed,
                reference_points=ref_points_input,
                spatial_shapes=spatial_shapes,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=level_start_index,
                encoder_hidden_states=encoder_hidden_states,
            )
            
            # Iterative Box Refinement (Updates the reference points for the next layer)
            predicted_corners = bbox_embed[idx](hidden_states)
            
            # Do not detach here yet, we need gradients to flow through the prediction for the loss
            reference_points = torch.sigmoid(predicted_corners + inverse_sigmoid(reference_points))
            
            intermediate_hidden_states.append(hidden_states)
            intermediate_ref_points.append(reference_points)
            
            # Detach for the next iteration's input to prevent gradient looping
            reference_points = reference_points.detach() 
            
        return torch.stack(intermediate_hidden_states), torch.stack(intermediate_ref_points)