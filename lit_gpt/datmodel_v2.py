'''
Shared causal attention
'''

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

FlashAttention2Available = RequirementCache("flash-attn>=2.0.0.post1")
RoPECache = Tuple[torch.Tensor, torch.Tensor]

class DualStreamGPTV2(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.n_embd % 2 == 0
        self.config = config

        # 1. Full Size Components
        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        # Use half_dim for the body streams
        self.wte = nn.Embedding(config.padded_vocab_size, config.n_embd)
        
        # 3. Optimized Body (Standard Blocks, shared weights for both streams)
        self.h_body = nn.ModuleList([
            Block(self.config) for _ in range(config.n_layer - 1)
        ])
        
        # 4. Last Layer (Fusion)
        self.h_last = FusedDualBlock(self.config)
        
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
               (name=="proj.weight" and isinstance(module, CausalSelfAttention)):
                nn.init.normal_(p, mean=0.0, std=1 / math.sqrt(self.config.n_embd)  /  n_layer)

    def forward(self, idx: torch.Tensor, max_seq_length: Optional[int] = None, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = idx.size()

        x_embd = self.wte(idx)
        
        # 1. Shifting logic
        zero_token = torch.zeros((B, 1), dtype=idx.dtype, device=idx.device)
        # idx_fwd = torch.cat([zero_token, idx[:, :-1]], dim=1)
        # x_fwd = self.wte(idx_fwd)
        idx_bwd_raw = torch.cat([idx[:, 1:], zero_token], dim=1)
        idx_bwd_flipped = torch.flip(idx_bwd_raw, dims=[1])
        
        # 2. Masking (stack along batch)
        if attn_mask is None:
            attn_mask = torch.ones((B, T), dtype=torch.bool, device=idx.device)
        valid_m = torch.ones((B, 1), dtype=torch.bool, device=idx.device)
        m_fwd = torch.cat([valid_m, attn_mask[:, :-1]], dim=1)
        m_bwd = torch.flip(torch.cat([attn_mask[:, 1:], valid_m], dim=1), dims=[1])
        
        # Combined Mask: [2*B, 1, 1, T] for SDPA
        combined_mask = torch.cat([m_fwd, m_bwd], dim=0).view(2*B, 1, 1, T)

        # 3. Embed and Batch-stack: [2*B, T, half_dim]
        x_bwd = self.wte(idx_bwd_flipped)
        x = torch.cat([x_fwd, x_bwd], dim=0)

        # 4. RoPE
        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(idx)
        cos, sin = self.rope_cache
        rope = (cos[:T], sin[:T])

        # 5. Process Body (Parallelized via batching)
        for block in self.h_body:
            x = block(x, rope, combined_mask)

        # 6. Fusion (Splits 2B -> B, concats to Full Dim)
        x_fused = self.h_last(x, rope, combined_mask)

        # 7. Head
        return self.lm_head(self.ln_f(x_fused))
    
    def reset_cache(self) -> None:
        if self.rope_cache is not None and self.rope_cache.device.type == "xla":
            self.rope_cache = None

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
    
class Block(nn.Module):
    """Standard Transformer Block processing [2*B, T, D]"""
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = LLaMAMLP(config) if config._mlp_class == "LLaMAMLP" else GptNeoxMLP(config)

    def forward(self, x: torch.Tensor, rope: RoPECache, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_1(x), rope, mask)
        x = x + self.mlp(self.norm_2(x))
        return x

class FusedDualBlock(nn.Module):
    """Transition layer: Processes 2*B streams, merges them, and runs wide MLP"""
    def __init__(self, config) -> None:
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        
        self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
        # MLP maps Concat(half, half) -> full_dim
        self.mlp = LLaMAMLP(config, input_dim = config.n_embd * 2) if config._mlp_class == "LLaMAMLP" else GptNeoxMLP(config, input_dim = config.n_embd * 2)

    def forward(self, x: torch.Tensor, rope: RoPECache, mask: torch.Tensor) -> torch.Tensor:
        # 1. Attn (still in 2*B mode)
        x = x + self.attn(self.norm_1(x), rope, mask)
        
        # 2. Split and Merge
        # x is [2*B, T, n_embd]
        x_fwd, x_bwd = torch.chunk(x, 2, dim=0) 
        x_bwd = torch.flip(x_bwd, dims=[1]) # Re-align BWD to FWD sequence
        
        # 3. Concatenate to Full Dim: [B, T, 2*n_embd]
        x_cat = torch.cat([self.norm_2(x_fwd), self.norm_2(x_bwd)], dim=-1)
        
        # 4. Wide Fusion MLP
        return self.mlp(x_cat)

class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(config.n_embd, shape, bias=config.bias)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

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
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.attn(x)

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size) # (B, T, n_query_groups, total_qkv, hs)
        # qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)

        # repeat k and v if necessary
        # Peiyuan: we do not need to do this as flash attention 2 already support GQA
        # if self.config.n_query_groups != 1:  # doing this would require a full kv cache with MQA (inefficient!)
        #     # for MHA this is a no-op
        #     k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
        #     v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)

        q = q.reshape(B,  T, -1, self.config.head_size)  # (B, T, nh_q, hs)
        k = k.reshape(B,  T, -1, self.config.head_size)  
        v = v.reshape(B,  T, -1, self.config.head_size)  

        cos, sin = rope

        # apply rope in fp32 significanly stabalize training
        # fused rope expect (batch_size, seqlen, nheads, headdim)
        q = apply_rotary_emb_func(q, cos, sin, False, True)
        k = apply_rotary_emb_func(k, cos, sin, False, True)
        
        combined_mask = None
        if mask is not None:
            padding_mask = mask.view(B, 1, 1, T)
            causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool)).view(1, 1, T, T)
            combined_mask = padding_mask & causal_mask

        y = self.scaled_dot_product_attention(q, k, v, mask=combined_mask)

        y = y.reshape(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)

        return y

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        scale = 1.0 / math.sqrt(self.config.head_size)
        
        if (
            FlashAttention2Available
            and mask is None
            and q.device.type == "cuda"
            and q.dtype in (torch.float16, torch.bfloat16)
        ):
            from flash_attn import flash_attn_func

            return flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale, causal=True)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if q.size() != k.size():
             k = k.repeat_interleave(q.shape[1]//k.shape[1], dim=1)
             v = v.repeat_interleave(q.shape[1]//v.shape[1], dim=1)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
        )
        return y.transpose(1, 2)

# Standard MLP Classes...
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