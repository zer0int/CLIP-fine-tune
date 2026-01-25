from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

# ============================================================
# Contrastive Losses
# ============================================================
class BaseContrastiveLoss(nn.Module):
    """
    Plain contrastive loss with label smoothing ONLY.
    """
    def __init__(self, temperature=0.07, smoothing=0.1):
        super().__init__()
        self.temperature = float(temperature)
        self.smoothing = float(smoothing)

    def forward(self, img_feats, txt_feats, return_parts: bool = False):
        img_feats = F.normalize(img_feats, p=2, dim=1)
        txt_feats = F.normalize(txt_feats, p=2, dim=1)

        logits = torch.matmul(img_feats, txt_feats.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)

        N = logits.size(0)
        if N == 1:
            smoothed_labels = torch.ones_like(logits)
        else:
            smoothed_labels = torch.full_like(logits, self.smoothing / (N - 1))
            smoothed_labels.scatter_(1, labels.unsqueeze(1), 1.0 - self.smoothing)

        log_probs = F.log_softmax(logits, dim=1)
        loss_img = -(smoothed_labels * log_probs).sum(dim=1).mean()

        log_probs_t = F.log_softmax(logits.t(), dim=1)
        loss_txt = -(smoothed_labels * log_probs_t).sum(dim=1).mean()

        base_loss = (loss_img + loss_txt) / 2.0
        total_loss = base_loss

        if return_parts:
            z = torch.zeros((), device=base_loss.device, dtype=base_loss.dtype)
            return total_loss, {
                "total_loss": total_loss.detach(),
                "base_loss": base_loss.detach(),
                "LUniform": z.detach(),
                "LAlign": z.detach(),
                "LXUniform": z.detach(),
                "w_uniform": 0.0,
                "w_align": 0.0,
                "w_xuniform": 0.0,
            }
        return total_loss

class GapContrastiveLoss(nn.Module):
    """
    Contrastive + Uniform, Align, XUniform components.
    Implemented after: arXiv:2405.18570v1
    Controlled by cfg w_* (scheduled per-epoch).
    """
    def __init__(
        self,
        temperature=0.07,
        smoothing=0.1,
        w_align=0.0,
        w_uniform=0.0,
        w_xuniform=0.0,
        eps=1e-8,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.smoothing = float(smoothing)
        self.w_align = float(w_align)
        self.w_uniform = float(w_uniform)
        self.w_xuniform = float(w_xuniform)
        self.eps = float(eps)

    def forward(self, img_feats, txt_feats, return_parts: bool = False):
        img_feats = F.normalize(img_feats, p=2, dim=1)
        txt_feats = F.normalize(txt_feats, p=2, dim=1)

        logits = torch.matmul(img_feats, txt_feats.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)

        N = logits.size(0)
        if N == 1:
            smoothed_labels = torch.ones_like(logits)
        else:
            smoothed_labels = torch.full_like(logits, self.smoothing / (N - 1))
            smoothed_labels.scatter_(1, labels.unsqueeze(1), 1.0 - self.smoothing)

        log_probs = F.log_softmax(logits, dim=1)
        loss_img = -(smoothed_labels * log_probs).sum(dim=1).mean()

        log_probs_t = F.log_softmax(logits.t(), dim=1)
        loss_txt = -(smoothed_labels * log_probs_t).sum(dim=1).mean()

        base_loss = (loss_img + loss_txt) / 2.0

        # Compute Uniform, Align, XUniform
        LUniform = torch.zeros((), device=base_loss.device, dtype=base_loss.dtype)
        LAlign = torch.zeros((), device=base_loss.device, dtype=base_loss.dtype)
        LXUniform = torch.zeros((), device=base_loss.device, dtype=base_loss.dtype)

        need_parts = return_parts or (self.w_uniform != 0.0) or (self.w_xuniform != 0.0) or (self.w_align != 0.0)
        if need_parts:
            sim_ii = img_feats @ img_feats.t()
            sim_tt = txt_feats @ txt_feats.t()
            sim_it = img_feats @ txt_feats.t()

            dist2_ii = (2.0 - 2.0 * sim_ii).clamp_min(0.0)
            dist2_tt = (2.0 - 2.0 * sim_tt).clamp_min(0.0)
            dist2_it = (2.0 - 2.0 * sim_it).clamp_min(0.0)

            # 1) Mask diagonal for ii/tt uniformity
            # 2) Normalize by number of valid pairs (use mean over selected pairs)
            # 3) Use upper triangle (i<j) for ii/tt to avoid double counting
            if N > 1:
                tri_mask = torch.triu(
                    torch.ones((N, N), device=dist2_ii.device, dtype=torch.bool),
                    diagonal=1
                )

                exp_ii = torch.exp(-2.0 * dist2_ii[tri_mask])
                exp_tt = torch.exp(-2.0 * dist2_tt[tri_mask])

                L_I_Uniform = torch.log(exp_ii.mean().clamp_min(self.eps))
                L_T_Uniform = torch.log(exp_tt.mean().clamp_min(self.eps))
                LUniform = 0.5 * (L_I_Uniform + L_T_Uniform)
            else:
                LUniform = torch.zeros((), device=base_loss.device, dtype=base_loss.dtype)

            # Cross-modal "XUniform": use only i!=j pairs and normalize by pair count (mean)
            if N > 1:
                offdiag_mask = ~torch.eye(N, device=dist2_it.device, dtype=torch.bool)
                exp_it = torch.exp(-2.0 * dist2_it[offdiag_mask])
                LXUniform = torch.log(exp_it.mean().clamp_min(self.eps))
            else:
                LXUniform = torch.zeros((), device=base_loss.device, dtype=base_loss.dtype)

            dist2_pos = dist2_it.diagonal()
            LAlign = torch.sqrt(dist2_pos + self.eps).mean()

        total_loss = base_loss + self.w_uniform * LUniform + self.w_align * LAlign + self.w_xuniform * LXUniform

        if return_parts:
            return total_loss, {
                "total_loss": total_loss.detach(),
                "base_loss": base_loss.detach(),
                "LUniform": LUniform.detach(),
                "LAlign": LAlign.detach(),
                "LXUniform": LXUniform.detach(),
                "w_uniform": float(self.w_uniform),
                "w_align": float(self.w_align),
                "w_xuniform": float(self.w_xuniform),
            }
        return total_loss


def apply_gap_loss_schedule(contrastive_loss: nn.Module, epoch_idx: int, cfg: TrainConfig):
    if not cfg.use_gap_schedule:
        return
    if not hasattr(contrastive_loss, "w_uniform"):
        return
    contrastive_loss.w_uniform = cfg.gap_w_uniform if epoch_idx >= cfg.gap_uniform_start_epoch else 0.0
    contrastive_loss.w_align   = cfg.gap_w_align   if epoch_idx >= cfg.gap_align_start_epoch   else 0.0
    contrastive_loss.w_xuniform= cfg.gap_w_xuniform if epoch_idx >= cfg.gap_xuniform_start_epoch else 0.0


# ============================================================
# KO Losses
# ============================================================
def k_proj_orthogonality_loss(model, selected_layers=None, lam=1.0):
    """
    Penalizes cosine similarity between heads' key projections for each expanded MLP feature.
    Only used when cfg.use_ko_config=True
    """
    if selected_layers is None:
        return torch.zeros((), device=next(model.parameters()).device)

    loss_total = 0.0
    count = 0

    for idx in selected_layers:
        block = model.visual.transformer.resblocks[idx]
        k_proj = block.attn.k_proj
        W = k_proj.weight  # [embed_dim, expanded_dim]
        embed_dim, expanded_dim = W.shape
        num_heads = block.attn.num_heads
        head_dim = embed_dim // num_heads

        W_heads = W.view(num_heads, head_dim, expanded_dim)
        W_heads_norm = F.normalize(W_heads, p=2, dim=1)

        sim = torch.einsum('ihf,jhf->ijf', W_heads_norm, W_heads_norm)  # [H,H,F]

        eye = torch.eye(num_heads, device=sim.device).unsqueeze(-1)  # [H,H,1]
        sim_no_diag = sim * (1 - eye)

        loss = (sim_no_diag ** 2).sum() / (num_heads * (num_heads - 1) * expanded_dim)
        loss_total += loss
        count += 1

    if count > 0:
        loss_total = loss_total / count
    loss_total = torch.as_tensor(loss_total, device=next(model.parameters()).device, dtype=torch.float32)
    return lam * loss_total

def decatt_loss(model, selected_layers=None, lam=1.0):
    """
    DeCAtt loss across selected layers of ViT visual transformer.
    See paper: "DeCAtt: Efficient Vision Transformers with Decorrelated Attention Heads"
    Only used when cfg.use_ko_config=True
    """
    device = next(model.parameters()).device
    if selected_layers is None:
        selected_layers = list(range(6))

    decatt_total = 0.0
    count = 0
    for idx in selected_layers:
        block = model.visual.transformer.resblocks[idx]
        x = getattr(block.attn, 'last_attn_output_per_head', None)
        if x is None:
            continue
        # x: [batch, heads, seq, head_dim]
        batch, heads, seq, head_dim = x.shape
        x_flat = x.permute(1, 0, 2, 3).contiguous().view(heads, batch * seq * head_dim)
        x_flat = F.normalize(x_flat, p=2, dim=1)
        c = torch.matmul(x_flat, x_flat.t()) / x_flat.shape[1]
        off_diag = c - torch.diag(torch.diag(c))
        loss = (off_diag ** 2).sum() / (heads * (heads - 1))
        decatt_total += loss
        count += 1

    if count > 0:
        decatt_total = decatt_total / count
    decatt_total = torch.as_tensor(decatt_total, device=device, dtype=torch.float32)
    return lam * decatt_total



# ============================================================
# Teacher loss
# ============================================================

# ============================================================
# REG delta + optional diagonal whitening (teacher-side)
# ============================================================
def reg_delta_whiten(
    reg_embed: torch.Tensor,
    reg_mean: torch.Tensor,
    reg_var: Optional[torch.Tensor],
    use_reg_whitening: bool = False,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    reg_embed: [B, D]
    reg_mean : [D] or [1, D]
    reg_var  : [D] or [1, D] (diagonal variance), or None if whitening disabled
    Returns:
      r_center            if use_reg_whitening=False
      r_center / sqrt(var) if use_reg_whitening=True
    """
    if reg_mean.dim() == 1:
        reg_mean = reg_mean.view(1, -1)
    r_center = reg_embed - reg_mean

    if not use_reg_whitening:
        return r_center

    if reg_var is None:
        raise ValueError("use_reg_whitening=True but reg_var is None")

    if reg_var.dim() == 1:
        reg_var = reg_var.view(1, -1)

    denom = torch.sqrt(reg_var + eps)
    return r_center / denom


def regression_consistency_loss(
    cls_embed: torch.Tensor,
    patch_embed: torch.Tensor,
    teacher: dict,
    normalize: bool = True,
    projector: Optional[nn.Module] = None,
    reg_embed: Optional[torch.Tensor] = None,
):
    patch_mean = teacher["patch_mean"].view(1, -1)
    cls_mean = teacher["cls_mean"].view(1, -1)
    W = teacher["W"]

    p_center = patch_embed - patch_mean

    if bool(teacher.get("is_reg_teacher", False)):
        if reg_embed is None:
            raise ValueError("teacher.is_reg_teacher=True but reg_embed is None")

        reg_mean = teacher["reg_mean"].view(1, -1)
        use_reg_whitening = bool(teacher.get("use_reg_whitening", False))
        reg_var = None
        if use_reg_whitening:
            reg_var = teacher["reg_var"].view(1, -1)

        r_in = reg_delta_whiten(
            reg_embed=reg_embed,
            reg_mean=reg_mean,
            reg_var=reg_var,
            use_reg_whitening=use_reg_whitening,
            eps=1e-4,
        )

        x_center = torch.cat([p_center, r_in], dim=1)
        c_center_hat = x_center @ W.T
    else:
        c_center_hat = p_center @ W.T

    c_hat = c_center_hat + cls_mean

    # handle ensemble projector by averaging per-projector losses
    if projector is not None and hasattr(projector, "project_all"):
        cls_all = projector.project_all(cls_embed.float())  # [K, B, d_out]
        hat_all = projector.project_all(c_hat.float())      # [K, B, d_out]
        embed_dim = cls_all.shape[-1]

        losses = []
        for k in range(cls_all.shape[0]):
            cls_use = cls_all[k]
            c_hat_use = hat_all[k]
            if normalize:
                cls_n = F.normalize(cls_use, dim=-1)
                c_hat_n = F.normalize(c_hat_use, dim=-1)
                losses.append(F.mse_loss(cls_n, c_hat_n) * embed_dim)
            else:
                losses.append(F.mse_loss(cls_use, c_hat_use) * embed_dim)
        return torch.stack(losses, dim=0).mean()

    # single-projector / no-projector path
    if projector is not None:
        cls_use = projector(cls_embed.float())
        c_hat_use = projector(c_hat.float())
    else:
        cls_use = cls_embed
        c_hat_use = c_hat

    embed_dim = cls_use.shape[-1]

    if normalize:
        cls_n = F.normalize(cls_use, dim=-1)
        c_hat_n = F.normalize(c_hat_use, dim=-1)
        loss = F.mse_loss(cls_n, c_hat_n) * embed_dim
    else:
        loss = F.mse_loss(cls_use, c_hat_use) * embed_dim

    return loss


# ============================================================
# Geometry-preserving term (VICReg: mean + variance + covariance)
# call this on a batch of image embeddings to fight rank-shedding / anisotropy
# ============================================================
def geometry_preserving_loss(
    x: torch.Tensor,
    var_floor: float = 0.02,
    eps: float = 1e-4,
    w_mean: float = 1.0,
    w_var: float = 1.0,
    w_cov: float = 1.0,
    normalize_input: bool = True,
) -> torch.Tensor:
    """
    x: [B, D] embeddings (typically image CLS embeddings).
    - mean loss keeps batch mean near 0 (prevents big center drift).
    - variance loss prevents per-dim collapse (std >= var_floor).
    - covariance loss discourages redundancy (off-diagonal covariance -> 0).
    """
    if x.dim() != 2:
        raise ValueError(f"geometry_preserving_loss expects [B,D], got {tuple(x.shape)}")

    if normalize_input:
        x = F.normalize(x, dim=-1)

    B, D = x.shape
    if B < 2:
        return x.new_tensor(0.0)

    # mean-centering
    mu = x.mean(dim=0, keepdim=True)
    x0 = x - mu

    loss_mean = (mu.pow(2).mean())

    # variance (std) floor
    var = x0.var(dim=0, unbiased=False) + eps
    std = torch.sqrt(var)
    loss_var = F.relu(var_floor - std).mean()

    # covariance (off-diagonal)
    cov = (x0.T @ x0) / float(B - 1)  # [D, D]
    off = cov - torch.diag(torch.diag(cov))
    #loss_cov = (off.pow(2).sum()) / float(D)
    loss_cov = off.pow(2).mean()

    return (w_mean * loss_mean) + (w_var * loss_var) + (w_cov * loss_cov)