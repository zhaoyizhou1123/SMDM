"""Full definition of a GPT NeoX Language Model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
"""
import math
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import Self
from flash_attn import flash_attn_func
from lit_gpt.config import Config
from xformers.ops import SwiGLU
from .fused_rotary_embedding import apply_rotary_emb_func
RoPECache = Tuple[torch.Tensor, torch.Tensor]
KVCache = Tuple[torch.Tensor, torch.Tensor]
FlashAttention2Available = RequirementCache("flash-attn>=2.0.0.post1")


class TriGPT(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size+1, config.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        
        # THREE-STREAM ARCHITECTURE SPECIFIC:
        # Learnable vector 'w' for the Query stream initialization (starts with no content)
        # self.query_token = nn.Parameter(torch.zeros(1, 1, config.n_embd))

        self.query_token_id = config.padded_vocab_size  # Using the last token id as the query token, which is also mask

        self.rope_cache: Optional[RoPECache] = None
        # self.mask_cache: Optional[torch.Tensor] = None
        # self.kv_caches: List[KVCache] = []

    def _init_weights(self, module: nn.Module, n_layer) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        # GPT-NeoX  https://arxiv.org/pdf/2204.06745.pdf
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        # GPT-NeoX       
        for name, p in module.named_parameters():
            if (name == "proj.weight" and isinstance(module, LLaMAMLP)) or (name == "w3.weight" and isinstance(module, SwiGLU) or (name=="proj.weight" and isinstance(module, CausalSelfAttention))):
                nn.init.normal_(p, mean=0.0, std=1 / math.sqrt(self.config.n_embd)  /  n_layer)
        

    def reset_cache(self) -> None:
        if self.rope_cache is not None and self.rope_cache.device.type == "xla":
            self.rope_cache = None

    def build_three_stream_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Constructs the attention mask for the 3-stream architecture.
        Returns a Boolean mask where True (1) indicates attention is allowed, 
        and False (0) indicates masked.
        
        Sequence layout: [Forward (H), Backward (B), Query (G)]
        Dimensions: 3*T x 2*T
        """
        # Initialize with False (0) -> No attention allowed by default
        mask = torch.zeros(3*T, 2*T, device=device, dtype=torch.bool)

        # 1. Forward Content Stream (H -> H): Standard Causal (j <= i)
        # Rows 0..T, Cols 0..T
        # We want the lower triangle including diagonal
        mask[:T, :T] = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))

        # 2. Backward Content Stream (B -> B): Reverse Causal (j >= i)
        # Rows T..2*T, Cols T..2*T
        # We want the upper triangle including diagonal
        mask[T:2*T, T:2*T] = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool))

        # 3. Query Stream (G -> Context)
        # Rows 2*T..3*T
        
        # Part A: G attends to H (Strict Past: j < i)
        # Cols 0..T. Lower triangle EXCLUDING diagonal
        mask[2*T:3*T, :T] = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=-1)
        
        # Part B: G attends to B (Strict Future: j > i)
        # Cols T..2*T. Upper triangle EXCLUDING diagonal
        mask[2*T:3*T, T:2*T] = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

        # Expand for batch and head dimensions: (1, 1, 3*T, 3*T)
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(
        self, 
        idx: torch.Tensor, 
        max_seq_length: Optional[int] = None, 
        input_pos: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            idx: Input tokens (B, T)
            use_three_stream: If True, activates the H/B/G stream logic.
        """
        B, T = idx.size()
        device = idx.device
        
        # 1. Prepare Embeddings
        # Content Embeddings for H and B
        content_emb = self.transformer.wte(idx) # (B, T, C)
        
        # Query Embeddings for G (all init to 'w')
        # g starts with no content, just the learnable vector w
        query_token = torch.full((1, 1), self.query_token_id, dtype=idx.dtype, device=idx.device)
        query_emb = self.transformer.wte(query_token)  # (1, 1, C)
        query_emb = query_emb.expand(B, T, -1) # (B, T, C)

        # 2. Concatenate inputs: [H, B, G] -> (B, 3*T, C)
        x = torch.cat([content_emb, content_emb, query_emb], dim=1)

        # 3. Handle Position Encodings (RoPE)
        # All three streams occupy positions 0...T-1 conceptually
        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(idx) # builds for max_block_size
        
        full_cos, full_sin = self.rope_cache
        
        # Construct explicit position indices: [0..T-1, 0..T-1, 0..T-1]
        range_t = torch.arange(T, device=device)
        three_stream_pos = torch.cat([range_t, range_t, range_t], dim=0) # (3*T,)
        
        # Select Rope embeddings for the concatenated sequence
        cos = full_cos.index_select(0, three_stream_pos)
        sin = full_sin.index_select(0, three_stream_pos)

        # 4. Construct the Three-Stream Mask
        mask = self.build_three_stream_mask(T, device)

        # 5. Forward Pass
        # Note: We treat 3*T as the sequence length. 
        # We disable Flash Attention causal mode inside blocks by passing explicit mask.
        max_seq_len_3x = T * 3 
        
        for block in self.transformer.h:
            # We pass the full concatenated sequence x (B, 3*T, C)
            # The block will apply the specific mask we built.
            x = block(x, (cos, sin), max_seq_len_3x, mask=mask)

        # 6. Final Layer Norm
        x = self.transformer.ln_f(x)

        # 7. Extract Query Stream (G) for prediction
        # The last T elements correspond to G
        g_out = x[:, 2*T : 3*T, :] # (B, T, C)

        # 8. Prediction
        logits = self.lm_head(g_out) # (B, T, Vocab)
        
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


class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        if not config.shared_attention_norm:
            self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = config.mlp_class(config)
        self.config = config
    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # n_1 = self.norm_1(x)
        # h = self.attn(n_1, rope, max_seq_length, mask, input_pos)       
        # x = x + h
        # x = x + self.mlp(self.norm_2(x))

        x = x + self.attn(self.norm_1(x), rope, max_seq_length, mask=mask)
        # print(f"Block.forward: after attn x.shape = {x.shape}")
        x = x + self.mlp(self.norm_2(x))
        # print(f"Block.forward: after mlp x.shape = {x.shape}")
        return x


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
    ) -> torch.Tensor:
        # print(f"CausalSelfAttention.forward: x.shape = {x.shape}")
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.attn(x)

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size) # (B, T, n_query_groups, total_qkv, hs)
        
        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)

        q = q.reshape(B,  T, -1, self.config.head_size)  # (B, T, nh_q, hs)
        k = k.reshape(B,  T, -1, self.config.head_size)  
        v = v.reshape(B,  T, -1, self.config.head_size)  

        cos, sin = rope

        # apply rope in fp32 significanly stabalize training
        # fused rope expect (batch_size, seqlen, nheads, headdim)
        q = apply_rotary_emb_func(q, cos, sin, False, True)
        k = apply_rotary_emb_func(k, cos, sin, False, True)

        if mask is not None and mask.shape[-1] < mask.shape[-2]:
            # mask shape is (..., 3T, 2T)
            valid_kv_len = mask.shape[-1] # 2T
            k = k[:, :valid_kv_len, ...]
            v = v[:, :valid_kv_len, ...]

        y = self.scaled_dot_product_attention(q, k, v, mask=mask)

        y = y.reshape(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)

        return y

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        scale = 1.0 / math.sqrt(self.config.head_size)
        
        # If mask is None and we are causal, use Flash Attention 2
        # However, for 3-stream architecture, mask is NOT None, so we fallback to manual SDPA
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
        
        # If mask is provided (Three-Stream), we assume is_causal=False in the backend
        # because the mask contains non-causal elements (like reverse causal or strict future)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
        )
        return y.transpose(1, 2)


class GptNeoxMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.gelu(x)
        return self.proj(x)


class LLaMAMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.swiglu = SwiGLU(config.n_embd,config.intermediate_size, bias=False, _pack_weights=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swiglu(x)


def build_rope_cache(
    seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000, condense_ratio: int = 1
) -> RoPECache:
    """Enhanced Transformer with Rotary Position Embedding."""
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta)

    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)

    # added by peiyuan to ensure same data type with q, k, to use fused rotary embedding
    if dtype == torch.bfloat16:
        return cos.bfloat16(), sin.bfloat16()
    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        return cos.half(), sin.half()
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    roped = (x * cos) + (rotated * sin)
    return roped.type_as(x)