import torch
import torch.nn.functional as F

from torch import nn
import numpy as np


class RoPE(nn.Module):
    def __init__(self, seq_len, head_dim, base=10000):
        super().__init__()
        self.head_dim = head_dim
        self.base = base

        pos = torch.arange(0, head_dim, 2)
        self.register_buffer('freqs', 1.0 / (base ** (pos / head_dim)), persistent=False)
        self._update_cos_sin_cache(seq_len)

    def _update_cos_sin_cache(self, seq_len):
        device = self.freqs.device
        positions = torch.arange(seq_len, device=device)
        angle_matrix = torch.outer(positions, self.freqs)
        cos = torch.cos(angle_matrix).unsqueeze(0).unsqueeze(0)
        sin = torch.sin(angle_matrix).unsqueeze(0).unsqueeze(0)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def forward(self, x):
        seq_len = x.shape[2]  
        if not hasattr(self, 'cos') or seq_len > self.cos.shape[2]:
            self._update_cos_sin_cache(seq_len)
            
        cos = self.cos[:, :, :seq_len, :]  
        sin = self.sin[:, :, :seq_len, :]
        
        x_even = x[..., 0::2] 
        x_odd = x[..., 1::2]

        x_even_rot = x_even * cos - x_odd * sin
        x_odd_rot = x_even * sin + x_odd * cos

        out = torch.empty_like(x)

        out[..., 0::2] = x_even_rot
        out[..., 1::2] = x_odd_rot
        return out


class GroupedQueryAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, seq_len=2048, window=0, dropout=0.1):
        
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.kv_dim = n_kv_heads * self.head_dim
        self.kv_repeats = self.n_heads//self.n_kv_heads
        self.window = window

        assert dim % n_heads == 0
        assert n_heads % n_kv_heads == 0

        self.rope = RoPE(seq_len, self.head_dim)


        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.kv_dim)
        self.v_proj = nn.Linear(self.dim, self.kv_dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            self.generate_mask(seq_len, window)
        )

    def generate_mask(self, seq_len, window=0):
        if window == 0:
            return torch.tril(torch.ones(seq_len, seq_len))
        else:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len))
            unattend = 1 - torch.triu(torch.ones(seq_len, seq_len), diagonal=-window)
            return causal_mask - unattend

    def forward(self, x):

        batch_size, seq_len, _, = x.shape
        
        q = self.q_proj(x).reshape(batch_size, seq_len, -1, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, seq_len, -1, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, seq_len, -1, self.head_dim)
        
        k = torch.repeat_interleave(k, self.kv_repeats, dim=2)
        v = torch.repeat_interleave(v, self.kv_repeats, dim=2)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        
        q = self.rope(q)
        k = self.rope(k)
        
        if self.window == 0:
            # Native FlashAttention for standard causal mask (O(1) memory)
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True
            )
        else:
            # Memory-efficient attention for sliding window custom mask
            if seq_len > self.mask.shape[0]:
                attn_mask = self.generate_mask(seq_len, self.window).to(x.device)
            else:
                attn_mask = self.mask[:seq_len, :seq_len]
                
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask.bool(),
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=False
            )
            
        out = out.transpose(1, 2)
        out = out.reshape(batch_size, seq_len, -1)
        out = self.out_proj(out)
        return out

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()

        self.dim = dim
        self.hidden_dim = hidden_dim

        self.in_linear = nn.Linear(dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU()
        )
        self.out_linear = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        proj = self.in_linear(x)
        gate = self.gate(x)
        out = self.dropout(proj * gate)
        return self.out_linear(out)


class TransformerBlock(nn.Module):

    def __init__(self, dim, hidden_dim, n_heads, n_kv_heads, seq_len=2048, window=0, dropout=0.1):
        super().__init__()

        self.attention = GroupedQueryAttention(dim, n_heads, n_kv_heads, seq_len=seq_len, window=window)
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        self.ffn = SwiGLU(dim, hidden_dim, dropout)

    def forward(self, x):
        attended = self.norm1(x)
        attended = self.attention(attended)
        attended = x + attended
        out = self.norm2(attended)
        out = self.ffn(out) + attended
        return out


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, dim, hidden_dim, n_heads, n_kv_heads, n_layers, seq_len=4096, local_att_every=-1, window=128, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.transformer = nn.ModuleList([])
        self.norm = nn.RMSNorm(dim)
        self.out = nn.Linear(dim, vocab_size, bias=False)
        self.out.weight = self.embedding.weight

        if local_att_every > 0:

            for n in range(1, n_layers+1):
                if n%local_att_every == 0:
                    self.transformer.append(
                        TransformerBlock(dim, hidden_dim, n_heads, n_kv_heads, seq_len=seq_len, window=window, dropout=dropout)
                    )
                else:
                    self.transformer.append(
                        TransformerBlock(dim, hidden_dim, n_heads, n_kv_heads, seq_len=seq_len, window=0, dropout=dropout)
                    )

        else:
            for n in range(1, n_layers+1):
                self.transformer.append(
                        TransformerBlock(dim, hidden_dim, n_heads, n_kv_heads, seq_len=seq_len, window=0, dropout=dropout)
                )

        # Apply custom weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.transformer:
            x = block(x)
        out = self.norm(x)
        out = self.out(out)
        return out



# dim = 64
# n_heads = 8
# n_kv_heads = 4
# batch_size = 5
# seq_len = 16


# x = torch.randint(0, 10000, (batch_size, seq_len)).to(torch.int64)
# # test transformer
# m = LanguageModel(10000, dim, dim*2, n_heads, n_kv_heads, 7, seq_len, local_att_every=3, window=4)

# out = m(x)

# print(out.shape)