import math
import copy
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import Self
from lit_gpt.config import Config
from xformers.ops import SwiGLU
from .fused_rotary_embedding import apply_rotary_emb_func

RoPECache = Tuple[torch.Tensor, torch.Tensor]
FlashAttention2Available = RequirementCache("flash-attn>=2.0.0.post1")

def get_mlp_class(mlp_class: str):
    if mlp_class == "GptNeoxMLP":
        return GptNeoxMLP
    elif mlp_class == "LLaMAMLP":
        return LLaMAMLP
    else:
        raise ValueError(f"Unsupported model_type {mlp_class} for MLP class.")


class DualStreamGPT(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        assert config.n_embd % 2 == 0, "n_embd must be divisible by 2 for split streams"
        assert config.n_head % 2 == 0, "n_head must be divisible by 2 for split streams"
        
        self.config = config
        self.half_dim = config.n_embd // 2

        # 1. Full Size Components (Embedding & Head)
        # We use standard size so we can tie weights if needed, and split/merge logic is cleaner.
        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.wte = nn.Embedding(config.padded_vocab_size, config.n_embd)
        
        # 2. Create "Half Config" for the Body Blocks
        # These blocks operate on n_embd // 2
        self.stream_config = copy.deepcopy(config)
        # self.stream_config.n_embd = self.half_dim
        self.stream_config.n_head = config.n_head // 2
        self.stream_config.n_query_groups = config.n_query_groups // 2
        # self.stream_config.intermediate_size = config.intermediate_size // 2
        
        # 3. Body Blocks (Layers 0 to N-2)
        # Operate on half_dim streams independently
        self.h_body = nn.ModuleList([
            DualStreamBlock(self.stream_config) for _ in range(config.n_layer - 1)
        ])
        
        # 4. Last Layer: FusedDualBlock
        # Takes half_dim streams -> Concats to full_dim -> MLP -> full_dim
        self.h_last = FusedDualBlock(self.stream_config)
        
        # 5. Final Norm (Full Dimension)
        self.ln_f = config.norm_class(config.n_embd, eps=config.norm_eps)
        
        self.rope_cache: Optional[RoPECache] = None

    def _init_weights(self, module: nn.Module, n_layer) -> None:
        # Standard Init logic
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
        for name, p in module.named_parameters():
            if (name == "proj.weight" and isinstance(module, LLaMAMLP)) or \
               (name == "w3.weight" and isinstance(module, SwiGLU)) or \
               (name=="proj.weight" and isinstance(module, CausalSelfAttentionHalf)):
                nn.init.normal_(p, mean=0.0, std=1 / math.sqrt(self.config.n_embd)  /  n_layer)

    def reset_cache(self) -> None:
        if self.rope_cache is not None and self.rope_cache.device.type == "xla":
            self.rope_cache = None

    def forward(
        self, 
        idx: torch.Tensor, 
        max_seq_length: Optional[int] = None, 
        input_pos: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None 
    ) -> torch.Tensor:
        
        B, T = idx.size()
        device = idx.device
        
        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(idx)
        
        cos, sin = self.rope_cache
        cos = cos[:T]
        sin = sin[:T]
        
        # -------------------------------------------------------------
        # 1. Input Shifting & Splitting
        # -------------------------------------------------------------
        
        # Zero token for padding shifts
        zero_token = torch.zeros((B, 1), dtype=idx.dtype, device=device)

        # FWD Input: [0, x_0, ..., x_{T-2}] -> Pos i sees x_{i-1}
        idx_fwd = torch.cat([zero_token, idx[:, :-1]], dim=1)
        
        # BWD Input: [x_1, ..., x_{T-1}, 0] -> Pos i sees x_{i+1}
        idx_bwd_raw = torch.cat([idx[:, 1:], zero_token], dim=1)
        
        # Masks
        if attn_mask is None:
            attn_mask = torch.ones((B, T), dtype=torch.bool, device=device)
        valid_mask = torch.ones((B, 1), dtype=torch.bool, device=device)
        
        mask_fwd = torch.cat([valid_mask, attn_mask[:, :-1]], dim=1)
        mask_bwd_raw = torch.cat([attn_mask[:, 1:], valid_mask], dim=1)
        
        # Flip BWD
        idx_bwd_flipped = torch.flip(idx_bwd_raw, dims=[1])
        mask_bwd_flipped = torch.flip(mask_bwd_raw, dims=[1])

        # Embed (Full Dim)
        x_fwd = self.wte(idx_fwd)           # [B, T, n_embd]
        x_bwd = self.wte(idx_bwd_flipped)   # [B, T, n_embd]

        # -------------------------------------------------------------
        # 2. Body Processing (Half Dim)
        # -------------------------------------------------------------
        for block in self.h_body:
            x_fwd, x_bwd = block(
                x_fwd, x_bwd, 
                (cos, sin), 
                max_seq_length,
                mask_fwd=mask_fwd, 
                mask_bwd=mask_bwd_flipped
            )

        # -------------------------------------------------------------
        # 3. Fused Processing (Concat -> Full Dim MLP)
        # -------------------------------------------------------------
        # Returns [B, T, n_embd]
        x_fused = self.h_last(
            x_fwd, x_bwd,
            (cos, sin),
            max_seq_length,
            mask_fwd=mask_fwd,
            mask_bwd=mask_bwd_flipped
        )

        # 4. Final Norm & Head
        x_out = self.ln_f(x_fused)
        logits = self.lm_head(x_out)

        return logits

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def build_rope_cache(self, idx: torch.Tensor) -> RoPECache:
        return build_rope_cache(
            seq_len=self.config.block_size,
            n_elem=int(self.config.rotary_percentage * self.config.head_size),
            dtype=torch.bfloat16,
            device=idx.device,
            condense_ratio=self.config.condense_ratio,
        )


class DualStreamBlock(nn.Module):
    """
    Operates on n_embd // 2.
    Uses 'stream_config' where n_embd and intermediate_size are halved.
    """
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn_fwd = CausalSelfAttentionHalf(config)
        self.attn_bwd = CausalSelfAttentionHalf(config)
        self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = get_mlp_class(config._mlp_class)(config) # Reduced size MLP
        self.config = config

    def forward(
        self,
        x_fwd: torch.Tensor,
        x_bwd: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask_fwd: Optional[torch.Tensor] = None,
        mask_bwd: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        x_fwd = x_fwd + self.attn_fwd(self.norm_1(x_fwd), rope, max_seq_length, mask=mask_fwd)
        x_fwd = x_fwd + self.mlp(self.norm_2(x_fwd))

        x_bwd = x_bwd + self.attn_bwd(self.norm_1(x_bwd), rope, max_seq_length, mask=mask_bwd)
        x_bwd = x_bwd + self.mlp(self.norm_2(x_bwd))

        return x_fwd, x_bwd


class FusedDualBlock(nn.Module):
    """
    Last Layer.
    Attention: Operates on Half Dim (stream_config).
    MLP: Operates on Full Dim (config).
    Logic: Concat(Half, Half) -> Full -> MLP(Full) -> Full.
    """
    def __init__(self, stream_config: Config) -> None:
        super().__init__()
        # Norms for the split streams
        self.norm_1 = stream_config.norm_class(stream_config.n_embd, eps=stream_config.norm_eps)
        
        # Attention on split streams
        self.attn_fwd = CausalSelfAttentionHalf(stream_config)
        self.attn_bwd = CausalSelfAttentionHalf(stream_config)
        
        # Norms before concatenation (or before MLP depending on PreNorm)
        # We need to norm the half-dim streams before concat? 
        # Usually PreNorm: x = x + MLP(Norm(x)).
        # Here: x_full = Concat(x_half_1, x_half_2).
        # x_out = MLP(Norm_Full(x_full)) ??
        # Or Concat(Norm(x1), Norm(x2))?
        # Let's use: Norm each stream, Concat, then MLP.
        self.norm_2 = stream_config.norm_class(stream_config.n_embd, eps=stream_config.norm_eps)
        
        # Full Size MLP (Restores to full n_embd)
        # We use the ORIGINAL config for the MLP (Full width)
        self.mlp = get_mlp_class(stream_config._mlp_class)(stream_config, input_dim=stream_config.n_embd * 2)
        self.config = stream_config # Save full config for parallel_residual check

    def forward(
        self,
        x_fwd: torch.Tensor,
        x_bwd: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask_fwd: Optional[torch.Tensor] = None,
        mask_bwd: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        # 1. Attention (Half Dim)
        x_fwd = x_fwd + self.attn_fwd(self.norm_1(x_fwd), rope, max_seq_length, mask=mask_fwd)
        x_bwd = x_bwd + self.attn_bwd(self.norm_1(x_bwd), rope, max_seq_length, mask=mask_bwd)

        # 2. Align BWD
        x_bwd_aligned = torch.flip(x_bwd, dims=[1])
        
        # 3. Norm & Concatenate
        # We apply the second norm (usually input to MLP) to the split streams individually
        n_fwd = self.norm_2(x_fwd)
        n_bwd = self.norm_2(x_bwd_aligned)
        
        # Concat: [B, T, d] + [B, T, d] -> [B, T, 2d]
        x_cat = torch.cat([n_fwd, n_bwd], dim=-1) 

        # 4. Full MLP
        # Maps d -> d (preserves size)
        x_out = self.mlp(x_cat)
        
        # Note: No residual connection here because dimensions changed (d/2 -> d)
        # The MLP acts as the fusion interface.
        return x_out


# --- Standard Components (Same as provided previously) ---

class CausalSelfAttentionHalf(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.head_size % 2 == 0, f"head_size {config.head_size} must be divisible by 2"
        self.head_size = config.head_size // 2
        shape = (config.n_head + 2 * config.n_query_groups) * self.head_size
        self.attn = nn.Linear(config.n_embd, shape, bias=config.bias)
        self.proj = nn.Linear(config.n_embd // 2, config.n_embd, bias=config.bias)
        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple] = None,
    ) -> torch.Tensor:
        B, T, C = x.size()
        assert C % 2 == 0, "Input embedding dimension must be even."
        qkv = self.attn(x)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2
        qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.head_size)
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)
        q = q.reshape(B,  T, -1, self.head_size)
        k = k.reshape(B,  T, -1, self.head_size)
        v = v.reshape(B,  T, -1, self.head_size)
        cos, sin = rope
        q = apply_rotary_emb_func(q, cos, sin, False, True)
        k = apply_rotary_emb_func(k, cos, sin, False, True)

        combined_mask = None
        if mask is not None:
            padding_mask = mask.view(B, 1, 1, T)
            causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool)).view(1, 1, T, T)
            combined_mask = padding_mask & causal_mask
        
        y = self.scaled_dot_product_attention(q, k, v, mask=combined_mask)
        y = y.reshape(B, T, C // 2)
        y = self.proj(y)
        return y

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        scale = 1.0 / math.sqrt(self.head_size)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if q.size() != k.size():
             k = k.repeat_interleave(q.shape[1]//k.shape[1], dim=1)
             v = v.repeat_interleave(q.shape[1]//v.shape[1], dim=1)

        if mask is not None:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=False
            )
        else:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, scale=scale, is_causal=True
            )
        return y.transpose(1, 2)

class GptNeoxMLP(nn.Module):
    def __init__(self, config: Config, input_dim: int = None) -> None:
        super().__init__()
        if input_dim is None:
            input_dim = config.n_embd
        self.fc = nn.Linear(input_dim, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.gelu(x)
        return self.proj(x)

class LLaMAMLP(nn.Module):
    def __init__(self, config: Config, input_dim: int = None) -> None:
        super().__init__()
        if input_dim is None:
            input_dim = config.n_embd
        self.swiglu = SwiGLU(input_dim, config.intermediate_size, out_features=config.n_embd, bias=False, _pack_weights=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swiglu(x)

def build_rope_cache(
    seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000, condense_ratio: int = 1
) -> RoPECache:
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device) / n_elem))
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio
    idx_theta = torch.outer(seq_idx, theta)
    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)
    if dtype == torch.bfloat16:
        return cos.bfloat16(), sin.bfloat16()
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        return cos.half(), sin.half()
    return cos, sin

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]
    x2 = x[..., head_size // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    roped = (x * cos) + (rotated * sin)
    return roped.type_as(x)