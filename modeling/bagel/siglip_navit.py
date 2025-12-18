# Copyright (c) 2024 The HuggingFace Inc. team.
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.

# Modified by AIDC-AI, 2025
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, 
# software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, 
# either express or implied. See the License for the specific language governing permissions and limitations under the License.


import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from typing import Any, Optional, Tuple, Union

from transformers.activations import ACT2FN
from modeling.siglip.configuration_siglip import SiglipVisionConfig as _SiglipVisionConfig
from modeling.siglip.modeling_siglip import SiglipAttention, SiglipSdpaAttention, SiglipPreTrainedModel
from transformers.utils import is_flash_attn_2_available
if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func


torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 4096
torch._dynamo.config.suppress_errors = True
flex_attention = torch.compile(flex_attention)


# 恢复 padded tensor
def to_padded(x, seqlens, batch_size, max_seqlen):
    xs = torch.split(x, seqlens)
    padded = torch.zeros(batch_size, max_seqlen, x.shape[-2], x.shape[-1],
                            dtype=x.dtype, device=x.device)
    for i, seq in enumerate(xs):
        padded[i, :seq.shape[0]] = seq
    return padded.transpose(1, 2).contiguous()


def flex_varlen_attention(q, k, v, cu_seqlens, max_seqlen, causal=False):
    """ 使用 flex_attention 实现变长注意力，接口对齐 flash_attn_varlen_func """
    seqlens = torch.diff(cu_seqlens).tolist()
    batch_size = len(seqlens)
    device = q.device

    q_pad = to_padded(q, seqlens, batch_size, max_seqlen)
    k_pad = to_padded(k, seqlens, batch_size, max_seqlen)
    v_pad = to_padded(v, seqlens, batch_size, max_seqlen)

    # 构造一个函数：只有当 q_pos 和 kv_pos 都在有效范围内时才允许 attention
    def padding_mask_mod(b, h, q_idx, kv_idx):
        # b: batch index, h: head index (可忽略), q_idx/kv_idx: 位置索引
        # 返回 True 表示允许连接
        valid_lengths = torch.tensor(seqlens, device=device)  # [100, 80, 120]
        q_in_range = q_idx < valid_lengths[b]      # q_idx < seqlens[b]
        kv_in_range = kv_idx < valid_lengths[b]    # kv_idx < seqlens[b]
        return q_in_range & kv_in_range

    # 构造 mask
    arange = torch.arange(max_seqlen, device=device)
    valid = arange.unsqueeze(0) < torch.tensor(seqlens, device=device).unsqueeze(1)  # (B, L)

    block_mask = create_block_mask(
        padding_mask_mod, B=batch_size, H=None, Q_LEN=max_seqlen, KV_LEN=max_seqlen, 
        device=device, BLOCK_SIZE=128, _compile=True
    )

    # 计算 attention
    out = flex_attention(q_pad, k_pad, v_pad, block_mask=block_mask).transpose(1, 2).contiguous()

    # flatten 回展平格式
    out_list = [out[i, :seqlens[i]] for i in range(batch_size)]
    return torch.cat(out_list, dim=0)


class SiglipVisionConfig(_SiglipVisionConfig):
    r"""
    This is the configuration class to store the configuration of a [`SiglipVisionModel`]. It is used to instantiate a
    Siglip vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the Siglip
    [google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input images.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    Example:

    ```python
    >>> from transformers import SiglipVisionConfig, SiglipVisionModel

    >>> # Initializing a SiglipVisionConfig with google/siglip-base-patch16-224 style configuration
    >>> configuration = SiglipVisionConfig()

    >>> # Initializing a SiglipVisionModel (with random weights) from the google/siglip-base-patch16-224 style configuration
    >>> model = SiglipVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "siglip_vision_model"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        rope=True,
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_channels=num_channels,
            image_size=image_size,
            patch_size=patch_size,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            attention_dropout=attention_dropout,
            **kwargs)
        
        self.rope = rope


class RotaryEmbedding2D(torch.nn.Module):
    def __init__(self, dim, max_h, max_w, base=10000):
        super().__init__()
        freq = torch.arange(0, dim, 2, dtype=torch.int64).float() / dim
        inv_freq = 1.0 / (base ** freq)

        grid_h = torch.arange(0, max_h)
        grid_h = grid_h.to(inv_freq.dtype)
        grid_h = grid_h[:, None].repeat(1, max_w)

        grid_w = torch.arange(0, max_w)
        grid_w = grid_w.to(inv_freq.dtype)
        grid_w = grid_w[None, :].repeat(max_h, 1)

        cos_h, sin_h = self._forward_one_side(grid_h, inv_freq)
        cos_w, sin_w = self._forward_one_side(grid_w, inv_freq)

        self.register_buffer("cos_h", cos_h)
        self.register_buffer("sin_h", sin_h)
        self.register_buffer("cos_w", cos_w)
        self.register_buffer("sin_w", sin_w)

    def _forward_one_side(self, grid, inv_freq):
        freqs = grid[..., None] * inv_freq[None, None, :]
        emb = torch.cat((freqs, freqs), dim=-1).flatten(0, 1)
        return emb.cos(), emb.sin()


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # unsqueeze due to the head dimension
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side**2
        self.num_positions = self.num_patches
        if not config.rope:
            self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def convert_conv2d_to_linear(self, config, meta=False):
        if meta:
            linear_patch_embedding = nn.Linear(
                config.num_channels * self.patch_size ** 2, self.embed_dim, bias=True, device='meta'
            )
        else:
            linear_patch_embedding = nn.Linear(
                config.num_channels * self.patch_size ** 2, self.embed_dim, bias=True
            )
        W = self.patch_embedding.weight.permute(0, 2, 3, 1).reshape(
            self.embed_dim, config.num_channels * self.patch_size ** 2
        )
        linear_patch_embedding.weight.data = W
        linear_patch_embedding.bias.data = self.patch_embedding.bias.data
        del self.patch_embedding
        self.patch_embedding = linear_patch_embedding

    def forward(
        self, 
        packed_pixel_values: torch.FloatTensor, 
        packed_flattened_position_ids: torch.LongTensor
    ) -> torch.Tensor:

        patch_embeds = self.patch_embedding(packed_pixel_values)
        if not self.config.rope:
            embeddings = patch_embeds + self.position_embedding(packed_flattened_position_ids)
        else:
            embeddings = patch_embeds
        return embeddings


class SiglipFlashAttention2(SiglipAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.IntTensor,
        max_seqlen: int,
        cos_h: torch.Tensor = None,
        sin_h: torch.Tensor = None,
        cos_w: torch.Tensor = None,
        sin_w: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:

        total_q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(total_q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(total_q_len, self.num_heads, self.head_dim)
        value_states = value_states.view(total_q_len, self.num_heads, self.head_dim)

        if self.config.rope:
            qh, qw = query_states[:, :, :self.head_dim // 2], query_states[:, :, self.head_dim // 2:] 
            kh, kw = key_states[:, :, :self.head_dim // 2], key_states[:, :, self.head_dim // 2:]
            qh, kh = apply_rotary_pos_emb(qh, kh, cos_h, sin_h)
            qw, kw = apply_rotary_pos_emb(qw, kw, cos_w, sin_w)
            query_states = torch.cat([qh, qw], dim=-1)
            key_states = torch.cat([kh, kw], dim=-1)

        attn_output = flash_attn_varlen_func(
            query_states.to(torch.bfloat16),
            key_states.to(torch.bfloat16),
            value_states.to(torch.bfloat16),
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=False,
        )

        attn_output = self.out_proj(attn_output.reshape(total_q_len, -1))
        return attn_output


class SiglipSdpaAttention2(SiglipSdpaAttention):
    
    is_causal = False

    def to_padded(self, tensor, seqlens, max_seqlen, batch_size, num_heads, head_dim):
        tensors = torch.split(tensor, seqlens)
        padded = torch.zeros(batch_size, max_seqlen, num_heads, head_dim,
                            dtype=tensor.dtype, device=tensor.device)
        for i, t in enumerate(tensors):
            padded[i, :t.shape[0]] = t
        return padded  # (B, L, H, D)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.IntTensor,
        max_seqlen: int,
        cos_h: torch.Tensor = None,
        sin_h: torch.Tensor = None,
        cos_w: torch.Tensor = None,
        sin_w: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> torch.Tensor:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "SiglipModel is using SiglipSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )

        total_q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(total_q_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(total_q_len, self.num_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(total_q_len, self.num_heads, self.head_dim)

        if self.config.rope:
            qh, qw = query_states[:, :, :self.head_dim // 2], query_states[:, :, self.head_dim // 2:] 
            kh, kw = key_states[:, :, :self.head_dim // 2], key_states[:, :, self.head_dim // 2:]
            qh, kh = apply_rotary_pos_emb(qh, kh, cos_h, sin_h)
            qw, kw = apply_rotary_pos_emb(qw, kw, cos_w, sin_w)
            query_states = torch.cat([qh, qw], dim=-1)
            key_states = torch.cat([kh, kw], dim=-1)

        # Step 1: 将展平的 q/k/v reshape 成 padded 形式 (B, L, H, D)
        seqlens = torch.diff(cu_seqlens).tolist()
        batch_size = len(seqlens)
        query_padded = self.to_padded(query_states, seqlens, max_seqlen, batch_size, self.num_heads, self.head_dim)
        key_padded   = self.to_padded(key_states,   seqlens, max_seqlen, batch_size, self.num_heads, self.head_dim)
        value_padded = self.to_padded(value_states, seqlens, max_seqlen, batch_size, self.num_heads, self.head_dim)

        # Step 2: 构造 attention mask
        # 方法A: 使用 bool mask (推荐，PyTorch 2.2+)
        attn_mask = torch.arange(max_seqlen, device=hidden_states.device).unsqueeze(0) < torch.tensor(seqlens, device=hidden_states.device).unsqueeze(1)
        # attn_mask: (B, L), True 表示 valid

        # 扩展为 (B, 1, L, L) 用于 attention
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
        attn_mask = attn_mask & attn_mask.transpose(-1, -2)  # (B, 1, L, L), 只有 valid × valid 位置为 True

        # Step 3: 调用 F.scaled_dot_product_attention
        # 注意：q/k/v 是 (B, L, H, D) -> 需转为 (B, H, L, D) 才符合 SDPA 接口
        query_sdpa = query_padded.transpose(1, 2)  # (B, H, L, D)
        key_sdpa   = key_padded.transpose(1, 2)
        value_sdpa = value_padded.transpose(1, 2)

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_sdpa.device.type == "cuda" and attn_mask is not None:
            query_sdpa = query_sdpa.contiguous()
            key_sdpa = key_sdpa.contiguous()
            value_sdpa = value_sdpa.contiguous()

        # 执行 attention
        attn_output_sdpa = torch.nn.functional.scaled_dot_product_attention(
            query_sdpa.to(torch.bfloat16),
            key_sdpa.to(torch.bfloat16),
            value_sdpa.to(torch.bfloat16),
            attn_mask=attn_mask,        # (B, 1, L, L), bool
            dropout_p=0.0,
            is_causal=False
        )  # 输出: (B, H, L, D)

        # 转回 (B, L, H, D)
        attn_output_sdpa = attn_output_sdpa.transpose(1, 2)

        # Step 4: flatten 回展平格式
        attn_output_sdpa_flat = torch.cat([
            attn_output_sdpa[i, :seqlens[i]] for i in range(batch_size)
        ], dim=0)  # (total_tokens, H, D)

        attn_output = self.out_proj(attn_output_sdpa_flat.reshape(total_q_len, -1))

        return attn_output


class SiglipFlexAttention(SiglipAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.IntTensor,
        max_seqlen: int,
        cos_h: torch.Tensor = None,
        sin_h: torch.Tensor = None,
        cos_w: torch.Tensor = None,
        sin_w: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:

        total_q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(total_q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(total_q_len, self.num_heads, self.head_dim)
        value_states = value_states.view(total_q_len, self.num_heads, self.head_dim)

        if self.config.rope:
            qh, qw = query_states[:, :, :self.head_dim // 2], query_states[:, :, self.head_dim // 2:] 
            kh, kw = key_states[:, :, :self.head_dim // 2], key_states[:, :, self.head_dim // 2:]
            qh, kh = apply_rotary_pos_emb(qh, kh, cos_h, sin_h)
            qw, kw = apply_rotary_pos_emb(qw, kw, cos_w, sin_w)
            query_states = torch.cat([qh, qw], dim=-1)
            key_states = torch.cat([kh, kw], dim=-1)

        attn_output = flex_varlen_attention(
            query_states.to(torch.bfloat16),
            key_states.to(torch.bfloat16),
            value_states.to(torch.bfloat16),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            causal=False,
        )

        attn_output = self.out_proj(attn_output.reshape(total_q_len, -1))
        return attn_output


class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        # if is_flash_attn_2_available():
        #     self.self_attn = SiglipFlashAttention2(config)
        # else:
        self.self_attn = SiglipSdpaAttention2(config)
        # self.self_attn = SiglipFlexAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.IntTensor,
        max_seqlen: int,
        cos_h: torch.Tensor = None,
        sin_h: torch.Tensor = None,
        cos_w: torch.Tensor = None,
        sin_w: torch.Tensor = None
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            cos_h=cos_h,
            sin_h=sin_h,
            cos_w=cos_w,
            sin_w=sin_w
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        cu_seqlens: torch.IntTensor,
        max_seqlen: int,
        cos_h: torch.Tensor = None,
        sin_h: torch.Tensor = None,
        cos_w: torch.Tensor = None,
        sin_w: torch.Tensor = None,
    ) -> torch.Tensor:

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, cu_seqlens, max_seqlen,
                                          cos_h=cos_h, sin_h=sin_h, cos_w=cos_w, sin_w=sin_w)

        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        if config.rope:
            max_size = config.image_size // config.patch_size
            dim_head = config.hidden_size // config.num_attention_heads
            self.rope = RotaryEmbedding2D(dim_head // 2, max_size, max_size)

        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        packed_pixel_values: torch.Tensor,
        packed_flattened_position_ids: torch.LongTensor,
        cu_seqlens: torch.IntTensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(
            packed_pixel_values=packed_pixel_values, 
            packed_flattened_position_ids=packed_flattened_position_ids
        )

        extra_inputs = {}
        if self.config.rope:
            extra_inputs.update(
                cos_h = self.rope.cos_h[packed_flattened_position_ids],
                sin_h = self.rope.sin_h[packed_flattened_position_ids],
                cos_w = self.rope.cos_w[packed_flattened_position_ids],
                sin_w = self.rope.sin_w[packed_flattened_position_ids]
            )

        last_hidden_state = self.encoder(
            inputs_embeds=hidden_states, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, 
            **extra_inputs
        )
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state


class SiglipVisionModel(SiglipPreTrainedModel):
    config_class = SiglipVisionConfig
    main_input_name = "packed_pixel_values"

    def __init__(self, config: SiglipVisionConfig):
        super().__init__(config)

        self.vision_model = SiglipVisionTransformer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        packed_pixel_values: torch.Tensor,
        packed_flattened_position_ids: torch.LongTensor,
        cu_seqlens: torch.IntTensor,
        max_seqlen: int,
    ) -> torch.Tensor:

        return self.vision_model(
            packed_pixel_values=packed_pixel_values,
            packed_flattened_position_ids=packed_flattened_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
