import torch
from einops import rearrange
from torch import Tensor


# def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor,pe_q = None, attention_mask = None) -> Tensor:
#     if pe_q is None:
#         q, k = apply_rope(q, k, pe) 
#         x = torch.nn.functional.scaled_dot_product_attention(q, k, v,attn_mask=attention_mask) 
#         x = rearrange(x, "B H L D -> B L (H D)")
#         return x
#     else: 
#         q, k = apply_rope_qk(q, k, pe_q, pe) 
#         x = torch.nn.functional.scaled_dot_product_attention(q, k, v,attn_mask=attention_mask)
#         x = rearrange(x, "B H L D -> B L (H D)")
#         return x

def attention(
    q: Tensor, 
    k: Tensor, 
    v: Tensor, 
    pe: Tensor,
    pe_q: Tensor = None, 
    attention_mask: Tensor = None,
    return_weights: bool = False  # ✅ 추가
) -> Tensor | tuple[Tensor, Tensor]:
    
    if pe_q is None:
        q, k = apply_rope(q, k, pe)
    else:
        q, k = apply_rope_qk(q, k, pe_q, pe)

    # ⚠️ torch.nn.functional.scaled_dot_product_attention supports return_attn_mask from PyTorch 2.0+
    if return_weights:
        # ⚠️ This requires PyTorch 2.0+ and the `is_causal=False` parameter
        out, attn_weights = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            return_attn=True
        )
    else:
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask
        )
        attn_weights = None

    out = rearrange(out, "B H L D -> B L (H D)")

    return (out, attn_weights) if return_weights else out


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim # dim =16 + 56 + 56 
    omega = 1.0 / (theta**scale) # 64 omega
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2) 
    xq_out = freqs_cis[:, :, :xq_.shape[2], :, :, 0] * xq_[..., 0] + freqs_cis[:, :, :xq_.shape[2], :, :, 1] * xq_[..., 1] 
    xk_out = freqs_cis[:, :, :xk_.shape[2], :, :, 0] * xk_[..., 0] + freqs_cis[:, :, :xk_.shape[2], :, :, 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

def apply_rope_qk(xq: Tensor, xk: Tensor, freqs_cis_q: Tensor,freqs_cis_k: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2) 
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2) 
    xq_out = freqs_cis_q[:, :, :xq_.shape[2], :, :, 0] * xq_[..., 0] + freqs_cis_q[:, :, :xq_.shape[2], :, :, 1] * xq_[..., 1]  
    xk_out = freqs_cis_k[:, :, :xk_.shape[2], :, :, 0] * xk_[..., 0] + freqs_cis_k[:, :, :xk_.shape[2], :, :, 1] * xk_[..., 1] 
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
