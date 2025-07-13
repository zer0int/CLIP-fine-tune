import os
from collections import OrderedDict
from typing import Tuple, Union, Optional
import numpy as np
import warnings
import torch
from torch import Tensor
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from torch.nn import functional as F

"""
Attention visualization code below, original author:
github.com/hila-chefer/Transformer-MM-Explainability
"""

# We define this function as _pad because it takes an argument
# named pad, which clobbers the recursive reference to the pad
# function needed for __torch_function__ support
#pad = F._pad

# This class exists solely for Transformer; it has an annotation stating
# that bias is never None, which appeases TorchScript

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
        
        
class _LinearWithBias(torch.nn.Linear):
    bias: Tensor

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, bias=True)


class GeometricLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GeometricLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Radial component
        self.r = nn.Parameter(torch.Tensor(out_features, 1))
        # Angular component
        self.theta = nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.theta, a=np.sqrt(5))
        fan_in = self.in_features
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.r, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        u = F.normalize(self.theta, p=2, dim=1)  # Normalize theta to get unit vector u
        output = F.linear(input, self.r * u)     # Geometric parameterization
        if self.bias is not None:
            output += self.bias
        return output


def head_dropout(attn_out, dropout_p):
    # attn_out: [batch, num_heads, tgt_len, head_dim]
    device = attn_out.device
    B, H, T, D = attn_out.shape
    # Generate [B, H, 1, 1] mask (one mask per head, per sample)
    keep_mask = (torch.rand(B, H, 1, 1, device=device) > dropout_p).float()
    # Renormalize to keep expectation
    attn_out = attn_out * keep_mask / (1.0 - dropout_p)
    return attn_out



def multi_head_attention_forward(query: Tensor,
                                 key: Tensor,
                                 value: Tensor,
                                 embed_dim_to_check: int,
                                 num_heads: int,
                                 in_proj_weight: Tensor,
                                 in_proj_bias: Tensor,
                                 bias_k: Optional[Tensor],
                                 bias_v: Optional[Tensor],
                                 add_zero_attn: bool,
                                 dropout_p: float,
                                 out_proj_weight: Tensor,
                                 out_proj_bias: Tensor,
                                 training: bool = True,
                                 key_padding_mask: Optional[Tensor] = None,
                                 need_weights: bool = True,
                                 collect_attn_features: bool = True,
                                 attn_mask: Optional[Tensor] = None,
                                 use_separate_proj_weight: bool = False,
                                 q_proj_weight: Optional[Tensor] = None,
                                 k_proj_weight: Optional[Tensor] = None,
                                 v_proj_weight: Optional[Tensor] = None,
                                 static_k: Optional[Tensor] = None,
                                 static_v: Optional[Tensor] = None,
                                 attention_probs_forward_hook = None,
                                 attention_probs_backwards_hook = None,
                                 ) -> Tuple[Tensor, Optional[Tensor]]:
    if not torch.jit.is_scripting():
        tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v,
                    out_proj_weight, out_proj_bias)
        if any([type(t) is not Tensor for t in tens_ops]) and F.has_torch_function(tens_ops):
            return F.handle_torch_function(
                multi_head_attention_forward, tens_ops, query, key, value,
                embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias,
                bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight,
                out_proj_bias, training=training, key_padding_mask=key_padding_mask,
                need_weights=need_weights, attn_mask=attn_mask,
                use_separate_proj_weight=use_separate_proj_weight,
                q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight,
                v_proj_weight=v_proj_weight, static_k=static_k, static_v=static_v)
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
        else:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if attn_mask is not None:
        assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
            attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, \
            'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float('-inf'))
        else:
            attn_output_weights += attn_mask


    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = F.softmax(
        attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

    if attention_probs_forward_hook is not None:
        if attn_output_weights.requires_grad:
            attention_probs_forward_hook(attn_output_weights)
    # Set backward hook only if provided
    if attention_probs_backwards_hook is not None:
        if attn_output_weights.requires_grad:
            attn_output_weights.register_hook(attention_probs_backwards_hook)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    
    attn_output_per_head = attn_output.view(N, num_heads, tgt_len, head_dim)
    last_attn_output_per_head = attn_output_per_head.detach()
        
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        attn_output_weights = attn_output_weights.view(N, num_heads, tgt_len, src_len)
        if collect_attn_features:
            return attn_output, attn_output_weights, attn_output_per_head
        else:
            return attn_output, attn_output_weights
    else:
        if collect_attn_features:
            return attn_output, attn_output_per_head
        else:
            return attn_output, None




class MultiheadAttention(nn.Module):
    """
    Patched MultiheadAttention with explicit Q/K/V projections as separate nn.Linear modules.
    """
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(
        self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # EXPLICIT Q/K/V LINEARS
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.out_proj = _LinearWithBias(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

        # === CUT HERE: Insert persistent buffer for per-head output ===
        self.last_attn_output_per_head = None  # <-- This is needed for DeCAtt loss extraction!
        # === END CUT HERE ===

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
            nn.init.constant_(self.k_proj.bias, 0.)
            nn.init.constant_(self.v_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self, query, key, value, key_padding_mask=None,
        need_weights=True, attn_mask=None,
        attention_probs_forward_hook=None, attention_probs_backwards_hook=None,
        last_attn_output_per_head = None, collect_attn_features=True, 
    ):
        # Shapes:
        # query: [L, N, E]
        # key:   [S, N, E]
        # value: [S, N, E]
        # E = embed_dim
        L, N, E = query.shape
        S = key.shape[0]

        # Compute Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        head_dim = self.head_dim
        num_heads = self.num_heads

        # scaling
        q = q * (head_dim ** -0.5)

        # reshape: [L, N, E] -> [N, L, num_heads, head_dim] -> [N * num_heads, L, head_dim]
        def reshape(x):
            # x: [seq, batch, embed_dim]
            x = x.permute(1, 0, 2)  # [batch, seq, embed_dim]
            new_shape = (x.shape[0], x.shape[1], num_heads, head_dim)
            x = x.view(*new_shape)  # [batch, seq, num_heads, head_dim]
            x = x.permute(0, 2, 1, 3)  # [batch, num_heads, seq, head_dim]
            return x.reshape(-1, x.shape[2], head_dim)  # [batch * num_heads, seq, head_dim]

        q = reshape(q)
        k = reshape(k)
        v = reshape(v)

        if attn_mask is not None:
            if attn_mask.dtype == torch.uint8:
                attn_mask = attn_mask.to(torch.bool)
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                pass
            else:
                raise RuntimeError("attn_mask has unsupported dimension")

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(torch.bool)

        # bias_k, bias_v
        if self.bias_k is not None and self.bias_v is not None:
            k = torch.cat([k, self.bias_k.repeat(k.size(0) // num_heads, 1, 1)], dim=1)
            v = torch.cat([v, self.bias_v.repeat(v.size(0) // num_heads, 1, 1)], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        if self.add_zero_attn:
            k = torch.cat([k, torch.zeros((k.size(0), 1, head_dim), dtype=k.dtype, device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros((v.size(0), 1, head_dim), dtype=v.dtype, device=v.device)], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        src_len = k.size(1)
        tgt_len = q.size(1)

        # attention
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        attn_output_weights = attn_output_weights.view(N, num_heads, tgt_len, src_len)
        if attn_mask is not None:
            attn_output_weights += attn_mask.unsqueeze(1)
        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        attn_output_weights = attn_output_weights.view(N * num_heads, tgt_len, src_len)
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

        if attention_probs_forward_hook is not None:
            attention_probs_forward_hook(attn_output_weights)
        # Set backward hook only if provided
        if attention_probs_backwards_hook is not None:
            attn_output_weights.register_hook(attention_probs_backwards_hook)

        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.view(N, num_heads, tgt_len, head_dim)
        
        def head_dropout(attn_out, dropout_p):
            # attn_out: [batch, num_heads, tgt_len, head_dim]
            device = attn_out.device
            B, H, T, D = attn_out.shape
            # Generate [B, H, 1, 1] mask (one mask per head, per sample)
            keep_mask = (torch.rand(B, H, 1, 1, device=device) > dropout_p).float()
            # Renormalize to keep expectation
            attn_out = attn_out * keep_mask / (1.0 - dropout_p)
            return attn_out
        
        attn_output = head_dropout(attn_output, dropout_p=0.2)
        attn_output_per_head = attn_output.view(N, num_heads, tgt_len, head_dim)

        # ======= THIS IS THE CRITICAL LINE FOR DeCAtt: =======
        if collect_attn_features:
            self.last_attn_output_per_head = attn_output_per_head.detach()
        # =====================================================

        attn_output = attn_output.permute(0, 2, 1, 3).reshape(N, tgt_len, E)
        attn_output = attn_output.permute(1, 0, 2)  # [tgt_len, batch, embed_dim]
        attn_output = self.out_proj(attn_output)

        if need_weights:
            attn_output_weights = attn_output_weights.view(N, num_heads, tgt_len, src_len)
            if collect_attn_features:
                return attn_output, attn_output_weights, attn_output_per_head
            else:
                return attn_output, attn_output_weights
        else:
            if collect_attn_features:
                return attn_output, attn_output_per_head
            else:
                return attn_output, None



class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", GeometricLinear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", GeometricLinear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.attn_probs = None
        self.attn_grad = None

    def set_attn_probs(self, attn_probs):
        # Defensive: attn_probs should be [batch, heads, seq, seq]
        if attn_probs.dim() == 3:  # [N * num_heads, seq, seq]
            n_heads = self.attn.num_heads
            N = attn_probs.shape[0] // n_heads
            attn_probs = attn_probs.view(N, n_heads, attn_probs.shape[1], attn_probs.shape[2])
        self.attn_probs = attn_probs.detach().cpu()

    def set_attn_grad(self, attn_grad):
        self.attn_grad = attn_grad

    def attention(self, x: torch.Tensor, need_weights=True):
        # Only set backward hook if gradients are enabled.
        use_backward_hook = torch.is_grad_enabled()
        attn_mask = None

        if self.attn_mask is not None:
            n_ctx = x.shape[0]
            attn_mask = self.attn_mask[..., -n_ctx:, -n_ctx:].to(dtype=x.dtype, device=x.device)
        
        # Select hooks according to grad mode
        attention_probs_backwards_hook = self.set_attn_grad if use_backward_hook else None

        return self.attn(
            x, x, x,
            need_weights=need_weights,
            attn_mask=attn_mask,
            attention_probs_forward_hook=self.set_attn_probs,
            attention_probs_backwards_hook=attention_probs_backwards_hook,
            collect_attn_features = True
        )[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        vision_heads = vision_width // 64
        self.visual = VisualTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            # Patched: Explicit Q/K/V projections
            nn.init.normal_(block.attn.q_proj.weight, std=attn_std)
            nn.init.normal_(block.attn.k_proj.weight, std=attn_std)
            nn.init.normal_(block.attn.v_proj.weight, std=attn_std)
            if block.attn.q_proj.bias is not None:
                nn.init.zeros_(block.attn.q_proj.bias)
            if block.attn.k_proj.bias is not None:
                nn.init.zeros_(block.attn.k_proj.bias)
            if block.attn.v_proj.bias is not None:
                nn.init.zeros_(block.attn.v_proj.bias)

            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            if block.attn.out_proj.bias is not None:
                nn.init.zeros_(block.attn.out_proj.bias)

            # Handle GeometricLinear layers
            if isinstance(block.mlp.c_fc, GeometricLinear):
                nn.init.normal_(block.mlp.c_fc.r, std=fc_std)
                nn.init.kaiming_uniform_(block.mlp.c_fc.theta, a=np.sqrt(5))
            else:
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)

            if isinstance(block.mlp.c_proj, GeometricLinear):
                nn.init.normal_(block.mlp.c_proj.r, std=proj_std)
                nn.init.kaiming_uniform_(block.mlp.c_proj.theta, a=np.sqrt(5))
            else:
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        n_ctx = text.shape[-1]
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding[:n_ctx].type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear, GeometricLinear)):
            if isinstance(l, GeometricLinear):
                l.r.data = l.r.data.half()
                l.theta.data = l.theta.data.half()
            else:
                l.weight.data = l.weight.data.half()
                if l.bias is not None:
                    l.bias.data = l.bias.data.half()

    def _convert_weights_to_fp16(l):
        if isinstance(l, MultiheadAttention):
            for attr in ["q_proj", "k_proj", "v_proj", "out_proj"]:
                module = getattr(l, attr, None)
                if module is not None and hasattr(module, "weight"):
                    module.weight.data = module.weight.data.half()
                    if module.bias is not None:
                        module.bias.data = module.bias.data.half()
            for attr in ["bias_k", "bias_v"]:
                tensor = getattr(l, attr, None)
                if tensor is not None:
                    tensor.data = tensor.data.half()


        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def adjust_state_dict(state_dict):
    new_state_dict = {}
    
    for key, value in state_dict.items():
        # Handle the conversion for GeometricLinear layers
        if "mlp.c_fc.weight" in key:
            base_key = key.replace("weight", "")
            new_state_dict[base_key + "r"] = torch.norm(value, dim=1, keepdim=True)
            new_state_dict[base_key + "theta"] = F.normalize(value, p=2, dim=1)
        elif "mlp.c_proj.weight" in key:
            base_key = key.replace("weight", "")
            new_state_dict[base_key + "r"] = torch.norm(value, dim=1, keepdim=True)
            new_state_dict[base_key + "theta"] = F.normalize(value, p=2, dim=1)
        else:
            new_state_dict[key] = value

    return new_state_dict

def convert_state_dict_inproj_to_qkv(state_dict, prefix=''):
    """Convert in_proj_weight/in_proj_bias to q_proj/k_proj/v_proj in the given state_dict."""
    out = {}
    for key, value in state_dict.items():
        if key.endswith('.attn.in_proj_weight'):
            # Split [3*D, D] into q/k/v
            D = value.shape[1]
            q = value[:D, :]
            k = value[D:2*D, :]
            v = value[2*D:, :]
            base = key[:-len('.in_proj_weight')]
            out[base + '.q_proj.weight'] = q
            out[base + '.k_proj.weight'] = k
            out[base + '.v_proj.weight'] = v
        elif key.endswith('.attn.in_proj_bias'):
            D = value.shape[0] // 3
            q = value[:D]
            k = value[D:2*D]
            v = value[2*D:]
            base = key[:-len('.in_proj_bias')]
            out[base + '.q_proj.bias'] = q
            out[base + '.k_proj.bias'] = k
            out[base + '.v_proj.bias'] = v
        else:
            out[key] = value
    return out


def build_model(state_dict: dict):
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    state_dict = convert_state_dict_inproj_to_qkv(state_dict)

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    state_dict = adjust_state_dict(state_dict)  # Adjust the state dictionary
    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
