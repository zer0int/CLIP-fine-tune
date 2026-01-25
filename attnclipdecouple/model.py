# model.py
# =============================================================================
# Add "trace" / "masked last-attn recompute" support for
# reproducing arXiv: 2505.05892v2 style experiments in CLIP:
#  - Return last-block attention weights (per-head), V/K/Q tensors, logits
#  - Compute patch-only / register-only CLS embeddings by masking *CLS row* in
#    the final self-attention (pre-softmax masking, renormalized)
#  - Provide token norms + derived implicit "register_mask" from high-norm patches
#  - Provide skip-vs-attn norm diagnostics (for the "skip dominance" analysis)
# =============================================================================
"""
Example: Call:

trace = model.encode_image(
    image_tensor,
    return_trace=True,
    register_threshold=70.0,
    max_registers=4,
    min_registers=1,
    cls_mask_includes_self=True,
    return_tokens=True,   # optional
)

Returns:
* `image_embedding_full`: standard CLIP image embedding
* `image_embedding_patch_only`: last-layer **CLS-row** attention masked to exclude implicit registers
* `image_embedding_reg_only`: last-layer **CLS-row** attention masked to include only implicit registers
* `register_mask`: your implicit register identification per image (high-norm patches, capped to 1–4 by default)
* `patch_token_norms`: norms used to define implicit registers
* `last_attn_probs`, `last_attn_logits`, `last_v`: what you need for attention-map faithfulness and “recompute” style probes
* `cls_skip_norm`, `cls_attn_norm`: last-block skip vs attention pathway magnitude
* optionally `tokens_pre_ln_post_full`: full token matrix pre-ln_post for additional diagnostics
"""

import os
from collections import OrderedDict
from typing import Tuple, Union, Optional, Dict, Any
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


class MultiheadAttention(nn.Module):
    """
    Patched MultiheadAttention with explicit Q/K/V projections as separate nn.Linear modules.

      - Optional capture of Q/K/V and pre-softmax logits for interpretability experiments.
      - Optional per-batch mask applied ONLY to the CLS query row (tgt index 0) over src tokens,
        implemented as a pre-softmax additive mask (renormalizes).
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

        # existing head ablation support
        self.head_mask = None

        # caches for tracing (populated only when capture=True)
        self.last_q: Optional[torch.Tensor] = None          # [B, H, T, D]
        self.last_k: Optional[torch.Tensor] = None          # [B, H, S, D]
        self.last_v: Optional[torch.Tensor] = None          # [B, H, S, D]
        self.last_logits: Optional[torch.Tensor] = None     # [B, H, T, S]
        self.last_probs: Optional[torch.Tensor] = None      # [B, H, T, S]

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
        self,
        query, key, value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        attention_probs_forward_hook=None,
        attention_probs_backwards_hook=None,
        # tracing + CLS-row masking
        capture: bool = False,
        cls_src_keep_mask: Optional[torch.Tensor] = None,  # [B, src_len] bool; only applied to tgt index 0
        cls_mask_includes_self: bool = True,               # if False, also disallow CLS->CLS when masking
    ):
        # Shapes:
        # query: [L, N, E]
        # key:   [S, N, E]
        # value: [S, N, E]
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

        def reshape(x_):
            # x_: [seq, batch, embed_dim] -> [batch*heads, seq, head_dim]
            x_ = x_.permute(1, 0, 2)  # [batch, seq, embed_dim]
            x_ = x_.view(x_.shape[0], x_.shape[1], num_heads, head_dim)  # [B, seq, H, D]
            x_ = x_.permute(0, 2, 1, 3)  # [B, H, seq, D]
            return x_.reshape(-1, x_.shape[2], head_dim)  # [B*H, seq, D]

        q_r = reshape(q)  # [B*H, L, D]
        k_r = reshape(k)  # [B*H, S, D]
        v_r = reshape(v)  # [B*H, S, D]

        # per-head masking by zeroing V for heads with mask=0
        if self.head_mask is not None:
            mask_flat = self.head_mask.to(v_r.device).float().repeat(N).view(N * num_heads, 1, 1)
            v_r = v_r * mask_flat

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
            k_r = torch.cat([k_r, self.bias_k.repeat(k_r.size(0) // num_heads, 1, 1)], dim=1)
            v_r = torch.cat([v_r, self.bias_v.repeat(v_r.size(0) // num_heads, 1, 1)], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        if self.add_zero_attn:
            k_r = torch.cat([k_r, torch.zeros((k_r.size(0), 1, head_dim), dtype=k_r.dtype, device=k_r.device)], dim=1)
            v_r = torch.cat([v_r, torch.zeros((v_r.size(0), 1, head_dim), dtype=v_r.dtype, device=v_r.device)], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        src_len = k_r.size(1)
        tgt_len = q_r.size(1)

        # attention logits: [B*H, T, S]
        attn_logits = torch.bmm(q_r, k_r.transpose(1, 2))
        attn_logits = attn_logits.view(N, num_heads, tgt_len, src_len)  # [B, H, T, S]

        # CLS-row src masking (pre-softmax, renormalizes)
        if cls_src_keep_mask is not None:
            if cls_src_keep_mask.dtype != torch.bool:
                cls_src_keep_mask = cls_src_keep_mask.to(torch.bool)
            if cls_src_keep_mask.shape != (N, src_len):
                raise ValueError(f"cls_src_keep_mask must be [B, src_len] == {(N, src_len)}, got {tuple(cls_src_keep_mask.shape)}")

            # Optionally disallow CLS->CLS (src index 0) when masking
            if not cls_mask_includes_self:
                cls_src_keep_mask = cls_src_keep_mask.clone()
                cls_src_keep_mask[:, 0] = False

            neg_large = torch.finfo(attn_logits.dtype).min
            # apply ONLY to tgt index 0 (CLS query)
            disallow = (~cls_src_keep_mask).view(N, 1, 1, src_len)  # [B,1,1,S]
            attn_logits[:, :, 0:1, :] = attn_logits[:, :, 0:1, :].masked_fill(disallow, neg_large)

        if attn_mask is not None:
            # attn_mask is additive in this implementation
            attn_logits = attn_logits + attn_mask.unsqueeze(1)  # [B,1,T,S]

        if key_padding_mask is not None:
            attn_logits = attn_logits.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # softmax
        attn_probs = F.softmax(attn_logits.view(N * num_heads, tgt_len, src_len), dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        if attention_probs_forward_hook is not None:
            attention_probs_forward_hook(attn_probs)

        if attention_probs_backwards_hook is not None and attn_probs.requires_grad:
            attn_probs.register_hook(attention_probs_backwards_hook)

        # attention output: [B*H, T, D]
        attn_output = torch.bmm(attn_probs, v_r)
        attn_output = attn_output.view(N, num_heads, tgt_len, head_dim)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(N, tgt_len, E)  # [B,T,E]
        attn_output = attn_output.permute(1, 0, 2)  # [T,B,E]
        attn_output = self.out_proj(attn_output)

        # capture tensors for experiments
        if capture:
            # reshape q/k/v to [B,H,T/D,S,D]
            q_c = q_r.view(N, num_heads, tgt_len, head_dim)
            k_c = k_r.view(N, num_heads, src_len, head_dim)
            v_c = v_r.view(N, num_heads, src_len, head_dim)
            self.last_q = q_c
            self.last_k = k_c
            self.last_v = v_c
            self.last_logits = attn_logits  # [B,H,T,S]
            self.last_probs = attn_probs.view(N, num_heads, tgt_len, src_len)

        if need_weights:
            attn_probs_out = attn_probs.view(N, num_heads, tgt_len, src_len)
            return attn_output, attn_probs_out  # NO AVERAGING
        else:
            return attn_output, None


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.attn_probs = None
        self.attn_grad = None


    def set_attn_probs(self, attn_probs):
        # attn_probs: [N*H, T, S] or [B,H,T,S]
        if attn_probs.dim() == 3:
            n_heads = self.attn.num_heads
            N = attn_probs.shape[0] // n_heads
            attn_probs = attn_probs.view(N, n_heads, attn_probs.shape[1], attn_probs.shape[2])
        self.attn_probs = attn_probs.detach().cpu()


    def set_attn_grad(self, attn_grad):
        self.attn_grad = attn_grad

    # allow returning weights + capture + CLS-row masking
    def attention(
        self,
        x: torch.Tensor,
        need_weights: bool = False,
        cls_src_keep_mask: Optional[torch.Tensor] = None,
        cls_mask_includes_self: bool = True,
        capture: bool = False
    ):
        use_backward_hook = torch.is_grad_enabled()
        attn_mask = None

        if self.attn_mask is not None:
            n_ctx = x.shape[0]
            attn_mask = self.attn_mask[..., -n_ctx:, -n_ctx:].to(dtype=x.dtype, device=x.device)

        attention_probs_backwards_hook = self.set_attn_grad if use_backward_hook else None

        attn_out, attn_w = self.attn(
            x, x, x,
            need_weights=need_weights,
            attn_mask=attn_mask,
            attention_probs_forward_hook=self.set_attn_probs,
            attention_probs_backwards_hook=attention_probs_backwards_hook,
            capture=capture,
            cls_src_keep_mask=cls_src_keep_mask,
            cls_mask_includes_self=cls_mask_includes_self
        )
        return attn_out, attn_w

    # forward optionally returns extra info (but remains backwards-compatible)
    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
        cls_src_keep_mask: Optional[torch.Tensor] = None,
        cls_mask_includes_self: bool = True,
        capture: bool = False
    ):
        ln1 = self.ln_1(x)
        attn_out, attn_w = self.attention(
            ln1,
            need_weights=return_attn,
            cls_src_keep_mask=cls_src_keep_mask,
            cls_mask_includes_self=cls_mask_includes_self,
            capture=capture
        )
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        if return_attn:
            return x, attn_w
        return x


class Transformer(nn.Module):
    """
    Switch from nn.Sequential to nn.ModuleList so we can:
      - run all but last block normally
      - re-run last block with custom CLS-row attention masks
      - optionally capture last-block attention tensors
    """
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor, capture_layers: Optional[set] = None):
        for i, blk in enumerate(self.resblocks):
            cap = (capture_layers is not None) and (i in capture_layers)
            x = blk(x, capture=cap)  # <-- pass capture flag
        return x

    def forward_until(self, x: torch.Tensor, layer_idx_exclusive: int):
        for i in range(layer_idx_exclusive):
            x = self.resblocks[i](x)
        return x


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

    @torch.no_grad()
    def _make_implicit_register_mask(
        self,
        patch_token_norms: torch.Tensor,   # [B, n_patches]
        register_threshold: float = 70.0,  # This is *only* the case for ViT-L!
        max_registers: Optional[int] = 4,
        min_registers: int = 1
    ) -> torch.Tensor:
        """
        Define implicit 'registers' as high-norm patch tokens.
        Returns bool mask [B, n_patches] where True = register.
        """
        B, P = patch_token_norms.shape
        mask = patch_token_norms > register_threshold

        if max_registers is not None:
            # cap registers per sample to max_registers by keeping highest norms among those above threshold
            out = torch.zeros_like(mask)
            for b in range(B):
                idx = torch.nonzero(mask[b], as_tuple=False).flatten()
                if idx.numel() == 0:
                    # fallback: top-min_registers by norm
                    topk = torch.topk(patch_token_norms[b], k=min_registers, largest=True).indices
                    out[b, topk] = True
                else:
                    k = min(max_registers, idx.numel())
                    topk = idx[torch.topk(patch_token_norms[b, idx], k=k, largest=True).indices]
                    out[b, topk] = True
            return out

        # ensure at least min_registers
        out = mask.clone()
        for b in range(B):
            if out[b].sum().item() < min_registers:
                topk = torch.topk(patch_token_norms[b], k=min_registers, largest=True).indices
                out[b, topk] = True
        return out

    def forward(
        self,
        x: torch.Tensor,
        # trace mode
        return_trace: bool = False,
        register_threshold: float = 70.0,
        max_registers: Optional[int] = 8,
        min_registers: int = 1,
        cls_mask_includes_self: bool = True,
        return_tokens: bool = False,
        capture_layers: Optional[set] = None,
    ):
        """
        Default (return_trace=False): identical behavior to original: returns projected CLS embedding.

        return_trace=True: returns dict with:
          - image_embedding_full: [B, D]
          - image_embedding_patch_only: [B, D]
          - image_embedding_reg_only: [B, D]
          - register_mask: [B, n_patches] bool (implicit regs)
          - patch_token_norms: [B, n_patches] (pre-ln_post, post-last-block)
          - last_attn_probs: [B, H, T, S] (post-softmax) from last block
          - last_v: [B, H, S, head_dim] (value vectors) from last block
          - cls_skip_norm / cls_attn_norm (pre-MLP residual split diagnostics, last block)
          - optionally tokens_pre_ln_post: [B, 1+n_patches, width] if return_tokens=True
        """
        # patchify + add CLS
        x = self.conv1(x)  # [B, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, width, n_patches]
        x = x.permute(0, 2, 1)  # [B, n_patches, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x],
            dim=1
        )  # [B, 1+n_patches, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        # to LND
        x = x.permute(1, 0, 2)  # [T,B,C]
        last_idx = self.transformer.layers - 1

        if not return_trace:
            x = self.transformer(x, capture_layers=capture_layers)  # [T,B,C]
            x_bt = x.permute(1, 0, 2)                               # [B,T,C]
            cls = self.ln_post(x_bt[:, 0, :])

            if self.proj is not None:
                cls = cls @ self.proj

            if return_tokens:
                return {
                    "image_embedding_full": cls,
                    "tokens_pre_ln_post_full": x_bt,  # [B,T,C]
                }
            return cls

        # TRACE PATH
        # run up to (but excluding) last block
        x_pre_last = self.transformer.forward_until(x, last_idx)  # [T,B,C]

        # full last block with capture
        last_block = self.transformer.resblocks[last_idx]

        # for skip-vs-attn norm diagnostics: need pre-attn residual input for CLS
        ln1_full = last_block.ln_1(x_pre_last)
        attn_out_full, attn_w_full = last_block.attn(
            ln1_full, ln1_full, ln1_full,
            need_weights=True,
            attn_mask=None,  # vision blocks have no causal mask
            capture=True,
            cls_src_keep_mask=None,
            cls_mask_includes_self=cls_mask_includes_self
        )
        x_full = x_pre_last + attn_out_full
        x_full = x_full + last_block.mlp(last_block.ln_2(x_full))  # [T,B,C]

        # token-space for norms/mask
        tokens_full = x_full.permute(1, 0, 2)  # [B,T,C]
        patch_token_norms = tokens_full[:, 1:, :].norm(dim=-1)  # [B, n_patches]

        register_mask = self._make_implicit_register_mask(
            patch_token_norms=patch_token_norms.detach(),
            register_threshold=register_threshold,
            max_registers=max_registers,
            min_registers=min_registers
        ).to(device=tokens_full.device)

        # build CLS-row keep masks over src_len = 1 + n_patches
        B, n_patches = register_mask.shape
        src_len = 1 + n_patches

        # keep patches only: keep CLS (index 0) + non-register patches
        keep_patch_only = torch.ones((B, src_len), dtype=torch.bool, device=tokens_full.device)
        keep_patch_only[:, 1:] = ~register_mask

        # keep regs only: keep CLS + register patches
        keep_reg_only = torch.ones((B, src_len), dtype=torch.bool, device=tokens_full.device)
        keep_reg_only[:, 1:] = register_mask

        # recompute last block with CLS-row masked attention (patch-only)
        ln1 = last_block.ln_1(x_pre_last)
        attn_out_patch, _ = last_block.attn(
            ln1, ln1, ln1,
            need_weights=False,
            attn_mask=None,
            capture=False,
            cls_src_keep_mask=keep_patch_only,
            cls_mask_includes_self=cls_mask_includes_self
        )
        x_patch = x_pre_last + attn_out_patch
        x_patch = x_patch + last_block.mlp(last_block.ln_2(x_patch))
        tokens_patch = x_patch.permute(1, 0, 2)

        # recompute last block with CLS-row masked attention (reg-only)
        attn_out_reg, _ = last_block.attn(
            ln1, ln1, ln1,
            need_weights=False,
            attn_mask=None,
            capture=False,
            cls_src_keep_mask=keep_reg_only,
            cls_mask_includes_self=cls_mask_includes_self
        )
        x_reg = x_pre_last + attn_out_reg
        x_reg = x_reg + last_block.mlp(last_block.ln_2(x_reg))
        tokens_reg = x_reg.permute(1, 0, 2)

        # final embeddings
        cls_full = self.ln_post(tokens_full[:, 0, :])
        cls_patch = self.ln_post(tokens_patch[:, 0, :])
        cls_reg = self.ln_post(tokens_reg[:, 0, :])

        if self.proj is not None:
            cls_full = cls_full @ self.proj
            cls_patch = cls_patch @ self.proj
            cls_reg = cls_reg @ self.proj

        # skip-vs-attn norms (last block, CLS only, pre-MLP)
        cls_skip = x_pre_last[0]          # [B,C]  (CLS token before attn residual add)
        cls_attn = attn_out_full[0]       # [B,C]  (CLS attention output)
        cls_skip_norm = cls_skip.norm(dim=-1)
        cls_attn_norm = cls_attn.norm(dim=-1)

        # last block captures (from MultiheadAttention)
        # NOTE: last_block.attn is MultiheadAttention; capture=True populated last_q/k/v/logits/probs
        last_attn_probs = last_block.attn.last_probs  # [B,H,T,S]
        last_v = last_block.attn.last_v               # [B,H,S,D]
        last_logits = last_block.attn.last_logits     # [B,H,T,S]

        out: Dict[str, Any] = {
            "image_embedding_full": cls_full,
            "image_embedding_patch_only": cls_patch,
            "image_embedding_reg_only": cls_reg,
            "register_mask": register_mask,
            "patch_token_norms": patch_token_norms,
            "last_attn_probs": last_attn_probs,
            "last_attn_logits": last_logits,
            "last_v": last_v,
            "cls_skip_norm": cls_skip_norm,
            "cls_attn_norm": cls_attn_norm,
        }
        if return_tokens:
            out["tokens_pre_ln_post_full"] = tokens_full  # [B,T,C]
        return out


class CLIP(nn.Module):
    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "use_positional_embedding_res"):
            self.use_positional_embedding_res = False
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
                 transformer_layers: int,
                 use_positional_embedding_res: bool = False,     # <-- LongCLIP: internal-only switch
                 longclip_keep_len: int = 20                     # <-- LongCLIP: Long-CLIP convention
                 ):
        super().__init__()

        self.context_length = context_length
        self.use_positional_embedding_res = bool(use_positional_embedding_res)  # <-- LongCLIP
        self.longclip_keep_len = int(longclip_keep_len)                         # <-- LongCLIP

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

        if self.use_positional_embedding_res:                           # <-- LongCLIP
            self.positional_embedding_res = nn.Parameter(torch.empty(self.context_length, transformer_width))

            # masks are deterministic and should not be trained; keep them off state_dict if possible
            mask1 = torch.zeros(self.context_length, 1, dtype=torch.float32)
            keep_len = min(self.longclip_keep_len, self.context_length)
            mask1[:keep_len, :] = 1.0
            mask2 = 1.0 - mask1

            try:
                self.register_buffer("mask1", mask1, persistent=False)  # <-- LongCLIP
                self.register_buffer("mask2", mask2, persistent=False)  # <-- LongCLIP
            except TypeError:
                # older torch without persistent=
                self.register_buffer("mask1", mask1)                    # <-- LongCLIP
                self.register_buffer("mask2", mask2)                    # <-- LongCLIP    

        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if self.use_positional_embedding_res:                           # <-- LongCLIP
            nn.init.normal_(self.positional_embedding_res, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
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
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            if block.mlp.c_fc.bias is not None:
                nn.init.zeros_(block.mlp.c_fc.bias)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
            if block.mlp.c_proj.bias is not None:
                nn.init.zeros_(block.mlp.c_proj.bias)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    # optional trace return for image encoder
    def encode_image(
        self,
        image,
        return_trace: bool = False,
        register_threshold: float = 70.0,
        max_registers: Optional[int] = 4,
        min_registers: int = 1,
        cls_mask_includes_self: bool = True,
        return_tokens: bool = False
    ):
        return self.visual(
            image.type(self.dtype),
            return_trace=return_trace,
            register_threshold=register_threshold,
            max_registers=max_registers,
            min_registers=min_registers,
            cls_mask_includes_self=cls_mask_includes_self,
            return_tokens=return_tokens
        )

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        if self.use_positional_embedding_res:               # <-- LongCLIP: Long-CLIP add (pos * mask1 + pos_res * mask2)
            pos = self.positional_embedding.to(device=x.device, dtype=x.dtype)
            posr = self.positional_embedding_res.to(device=x.device, dtype=x.dtype)
            m1 = self.mask1.to(device=x.device, dtype=x.dtype)
            m2 = self.mask2.to(device=x.device, dtype=x.dtype)
            x = x + pos * m1 + posr * m2
        else:
            x = x + self.positional_embedding.to(device=x.device, dtype=x.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image, text):
        image_features = self.encode_image(image, return_trace=False)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""
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


def convert_state_dict_inproj_to_qkv(state_dict, prefix=''):
    """Convert in_proj_weight/in_proj_bias to q_proj/k_proj/v_proj in the given state_dict."""
    out = {}
    for key, value in state_dict.items():
        if key.endswith('.attn.in_proj_weight'):
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

    use_positional_embedding_res = ("positional_embedding_res" in state_dict)       # <--- Long-CLIP
    # -> sanity check when it IS present:
    if use_positional_embedding_res:
        pe = state_dict["positional_embedding"]
        per = state_dict["positional_embedding_res"]
        if pe.shape != per.shape:
            raise ValueError(
                f"positional_embedding_res shape mismatch: positional_embedding={tuple(pe.shape)} "
                f"vs positional_embedding_res={tuple(per.shape)}"
            )
    else:
        # If not using Long-CLIP, drop unexpected keys so strict loading works
        state_dict.pop("positional_embedding_res", None)

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        use_positional_embedding_res=use_positional_embedding_res
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict, strict=True)
    return model.eval()