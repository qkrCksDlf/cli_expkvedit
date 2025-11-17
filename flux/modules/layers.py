import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn

from flux.math import attention, rope,apply_rope

import os

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import attn_maps, save_attention_maps

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def overlay_attention_map(
    id,
    t,
    attn_weights: torch.Tensor,
    q_idx: int,
    h_idx: int = 0,
    img_size: tuple = (48, 32),
    base_image: Image.Image = None,  # PIL 이미지
    base_image_path: str = None,     # 경로를 줄 수도 있음
    save_path: str = "attn_overlay.png",
    cmap: str = "jet",
    alpha: float = 0.5,              # heatmap 투명도
    resize_to_base: bool = True      # 해상도 맞출지
):
    """
    Attention map을 원본 이미지 위에 오버레이합니다.

    - attn_weights: Tensor [B, H, Q, K]
    - q_idx: 시각화할 query index
    - h_idx: head index
    - img_size: attention map size (K = H x W)
    - base_image: PIL.Image (원본 이미지)
    - base_image_path: 이미지 경로 (위에 없으면 경로로 로드)
    - save_path: 저장 경로
    """

    assert base_image or base_image_path, "base_image or base_image_path must be provided"

    if base_image is None:
        base_image = Image.open(base_image_path).convert("RGB")

    # 1. attention map 얻기
    attn_map = attn_weights[0, h_idx, q_idx].detach().cpu().float().numpy()
    attn_map = np.squeeze(attn_map)

    # 2. reshape to 2D
    attn_map = attn_map.reshape(img_size)

    # 3. 정규화
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    # 4. resize heatmap to match base image
    base_np = np.array(base_image)
    if resize_to_base:
        attn_map = cv2.resize(attn_map, (base_np.shape[1], base_np.shape[0]))

    # 5. colormap 적용 (e.g., jet)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * attn_map), getattr(cv2, f'COLORMAP_{cmap.upper()}'))

    # 6. overlay
    overlayed = cv2.addWeighted(base_np, 1.0, heatmap_color, alpha, 0)
    save_path = f"attn_overlay/{id}and{t}.png"
    # 7. 저장
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(overlayed).save(save_path)
    print(f"📸 Attention overlay saved to: {save_path}")

def save_cross_attention_map(
    attn_weights: torch.Tensor,
    info: dict,
    txt_len: int,
    img_len: int,
    layer_tag: str,
    inp_target_s=None,  # 시그니처만 유지, 실제로는 안 씀
):
    """
    attn_weights: [B, H, Q, K]  (Q = txt_len + img_len, K = txt_len + img_len)
    txt_len: 텍스트 토큰 개수
    img_len: 이미지 토큰 개수
    layer_tag: "MB" / "SB" 등 레이어 구분용 문자열

    info:
        - 필수:
            - id: 샘플 식별자
            - t: timestep (float 또는 tensor)
        - 선택:
            - img_size = (H, W)
            - save_attn (bool)
            - token_list = ["▁", "a", "▁dog", ...]
            - q_token = "▁dog"  # 문자열 토큰 (토크나이저에서 나온 그대로)
    """

    # 0. 저장 플래그 없으면 바로 리턴
    if not info.get("save_attn", False):
        return

    B, H, Q, K = attn_weights.shape

    # 1. 길이 보정
    txt_len = min(txt_len, Q)
    max_img_len = K - txt_len
    if max_img_len <= 0:
        return
    img_len = min(img_len, max_img_len)

    # 2. text→image cross-attn만 추출: [B, H, txt_len, img_len]
    attn_text_to_img = attn_weights[:, :, :txt_len, txt_len:txt_len + img_len]

    # batch 0, head 평균 → [txt_len, img_len]
    attn_mean = attn_text_to_img[0].mean(dim=0)  # [txt_len, img_len]

    # ============================================
    # 3) q_token / token_list 기반으로 토큰 인덱스 결정
    #    → 못 찾으면 "아무것도 저장 안 함"
    # ============================================
    token_list = info.get("token_list", None)
    q_token = info.get("q_token", None)

    if token_list is None or q_token is None:
        print("⚠️ [save_cross_attention_map] token_list 또는 q_token 없음. 저장 스킵.")
        return

    # 문자열 완전 일치만 허용 (정규화 없이)
    token_indices = [i for i, tok in enumerate(token_list) if tok == q_token]

    if not token_indices:
        print(f"⚠️ [save_cross_attention_map] q_token={q_token} not found in token_list. 저장 스킵.")
        return

    token_indices = sorted(set(token_indices))
    print(f"✅ [save_cross_attention_map] q_token={q_token} -> indices={token_indices}")

    # ============================================
    # 4) img_len에서 자동으로 (img_h, img_w) 찾기
    #    - info['img_size']가 정확하면 그거 우선 사용
    #    - 아니면 img_len의 약수 중 sqrt에 가장 가까운 조합 사용
    # ============================================
    import math as _math

    img_h, img_w = info.get("img_size", (0, 0))
    if img_h * img_w != img_len:
        # 자동 factorization
        h = int(_math.sqrt(img_len))
        if h < 1:
            h = 1
        # sqrt에서 내려가면서 약수 찾기
        while h > 1 and img_len % h != 0:
            h -= 1
        w = img_len // h
        img_h, img_w = h, w

    from PIL import Image
    import numpy as _np
    import os as _os
    import torch as _torch

    attn_id = info.get("id", "unknown")
    attn_t = info.get("t", 0)

    # t를 float로 정리
    if isinstance(attn_t, _torch.Tensor):
        attn_t_val = float(attn_t.detach().cpu().item())
    else:
        attn_t_val = float(attn_t)

    # t 문자열: 소수 3자리까지만, 폴더 이름용 '.' → '_'
    attn_t_str = f"{attn_t_val:.3f}".replace(".", "_")

    # 저장 경로: attn_token_maps/{id}/{layer_tag}/t_{t값}/
    out_dir = _os.path.join(
        "attn_token_maps",
        str(attn_id),
        str(layer_tag),
        f"t_{attn_t_str}",
    )
    _os.makedirs(out_dir, exist_ok=True)

    # ============================================
    # 5) 선택된 토큰들만 저장
    # ============================================
    for tok_idx in token_indices:
        if tok_idx >= txt_len:
            continue

        vec = attn_mean[tok_idx]  # [img_len]
        vec = vec.detach().cpu().float().numpy()

        # 0~1 정규화
        vmin, vmax = vec.min(), vec.max()
        if vmax > vmin:
            vec = (vec - vmin) / (vmax - vmin)
        else:
            vec = _np.zeros_like(vec)

        # [img_h, img_w] 로 reshape
        heatmap = vec.reshape(img_h, img_w)

        # 0~255 uint8 그레이스케일
        img_arr = (heatmap * 255.0).clip(0, 255).astype("uint8")
        pil_img = Image.fromarray(img_arr, mode="L")

        # 보기 좋게 x4 업스케일
        scale = 4
        pil_img = pil_img.resize((img_w * scale, img_h * scale), resample=Image.NEAREST)
        pil_img = pil_img.convert("RGB")

        save_path = _os.path.join(
            out_dir,
            f"{attn_id}_{layer_tag}_t{attn_t_val:.4f}_tok{tok_idx}.png",
        )
        pil_img.save(save_path)

    print(
        f"✅ [save_cross_attention_map] saved {len(token_indices)} token maps "
        f"({img_h}x{img_w}) to {out_dir} for id={attn_id}, layer={layer_tag}, t={attn_t_val:.4f}"
    )







class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads) 

        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)
        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2) 
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)
        # import pdb;pdb.set_trace()
        attn = attention(q, k, v, pe=pe)
 
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        return img, txt
class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x

class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


class DoubleStreamBlock_kv(DoubleStreamBlock):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__(hidden_size, num_heads, mlp_ratio, qkv_bias)

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, info, info_s, zt_r, inp_target_s) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)
        print(img.shape, zt_r.shape)
        
        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads) 

        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)
        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)   
        
        feature_k_name = str(info['t']) + '_' + str(info['id']) + '_' + 'MB' + '_' + 'K'
        feature_v_name = str(info['t']) + '_' + str(info['id']) + '_' + 'MB' + '_' + 'V'
        if info['inverse']:
            info['feature'][feature_k_name] = img_k.cpu()
            info['feature'][feature_v_name] = img_v.cpu()
            q = torch.cat((txt_q, img_q), dim=2)
            k = torch.cat((txt_k, img_k), dim=2)
            v = torch.cat((txt_v, img_v), dim=2)
            if 'attention_mask' in info:
                attn = attention(q, k, v, pe=pe,attention_mask=info['attention_mask'])
            else:
                attn = attention(q, k, v, pe=pe)
    
        else:
            img_mod1, img_mod2 = self.img_mod(vec)
            txt_mod1, txt_mod2 = self.txt_mod(vec)
            # prepare reference image for attention
            img_modulated_r = self.img_norm1(zt_r)
            img_modulated_r = (1 + img_mod1.scale) * img_modulated_r + img_mod1.shift
            img_qkv_r = self.img_attn.qkv(img_modulated_r)
            img_q_r, img_k_r, img_v_r = rearrange(img_qkv_r, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)          
            img_q_r, img_k_r = self.img_attn.norm(img_q_r, img_k_r, img_v_r)
            

            source_img_k = info['feature'][feature_k_name].to(img.device) #레퍼런스
            source_img_v = info['feature'][feature_v_name].to(img.device) #레퍼런스
            source_img_k_s = info_s['feature'][feature_k_name].to(img.device) #소스
            source_img_v_s = info_s['feature'][feature_v_name].to(img.device) #소스
            #원본
            mask_indices = info['mask_indices'] 
            # key_list_s = list(info_s['feature'].keys())
            
            # key_index = -1 # 기본값 (못 찾음)
            # try:
            #     # 2. 현재 키 이름(feature_k_name)의 인덱스를 찾습니다.
            #     key_index = key_list_s.index(feature_k_name)
            # except ValueError:
            #     # 딕셔너리에 해당 키가 없으면, 인덱스를 찾을 수 없습니다.
            #     pass
                
            # if key_index <= 4 and info['t']>0.4:
            #     # [조건 만족] K,V 주입
            #     print(f"Index {key_index} >= 4. Injecting K/V for {feature_k_name}") # (디버깅용)
            #     source_img_k_s[:, :, mask_indices, ...] = source_img_k[:, :, mask_indices, ...]
            #     source_img_k_v[:, :, mask_indices, ...] = source_img_v[:, :, mask_indices, ...]
            
            # else:
            #     # [조건 불만족] K,V 주입 대신 Self-Attention 수행
            #     print(f"Index {key_index} < 4. Using Self-Attention for {feature_k_name}") # (디버깅용)
            #     source_img_k_s[:, :, mask_indices, ...] = img_k
            #     source_img_v_s[:, :, mask_indices, ...] = img_v


            source_img_k_s[:, :, mask_indices, ...] = img_k
            source_img_v_s[:, :, mask_indices, ...] = img_v

                
            q = torch.cat((txt_q, img_q), dim=2) #소스이미지
            k = torch.cat((txt_k, source_img_k_s), dim=2) 
            v = torch.cat((txt_v, source_img_v_s), dim=2)

            #attn = attention(q, k, v, pe=pe, pe_q = info['pe_mask'],attention_mask=info['attention_scale'])
            attn, attn_weights = attention(q, k, v, pe=pe, pe_q = info['pe_mask'],attention_mask=info['attention_scale'], return_weights=True)
            txt_len = txt.shape[1]
            img_len = img.shape[1]
            attn_text_to_img = attn_weights[:, :, :txt_len, txt_len:txt_len+img_len]

            if info.get("track_cross", False):
                info["tracker"].add(attn_text_to_img)

        
            # === 여기서 cross-attention map 저장 ===

            # save_cross_attention_map(
            #     attn_weights=attn_weights,
            #     info=info,
            #     txt_len=txt_len,
            #     img_len=img_len,
            #     layer_tag="MB",
            #     inp_target_s=inp_target_s,
            # )

        
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        return img, txt
    
class SingleStreamBlock_kv(SingleStreamBlock):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__(hidden_size, num_heads, mlp_ratio, qk_scale)

    def forward(self,x: Tensor, vec: Tensor, pe: Tensor, info, info_s, zt_r, inp_target_s) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        img_k = k[:, :, 512:, ...]
        img_v = v[:, :, 512:, ...]
        
        txt_k = k[:, :, :512, ...]
        txt_v = v[:, :, :512, ...]
    

        feature_k_name = str(info['t']) + '_' + str(info['id']) + '_' + 'SB' + '_' + 'K'
        feature_v_name = str(info['t']) + '_' + str(info['id']) + '_' + 'SB' + '_' + 'V'


        
        if info['inverse']:
            info['feature'][feature_k_name] = img_k.cpu()
            info['feature'][feature_v_name] = img_v.cpu()
            if 'attention_mask' in info:
                attn = attention(q, k, v, pe=pe,attention_mask=info['attention_mask'])
            else:
                attn = attention(q, k, v, pe=pe)
            
        else:

            feature_keys = list(info['feature'].keys())
            feature_k_index = feature_keys.index(feature_k_name)
            
            
            source_img_k = info['feature'][feature_k_name].to(x.device) #레퍼런스
            source_img_v = info['feature'][feature_v_name].to(x.device) #레퍼런스

            source_img_k_s = info_s['feature'][feature_k_name].to(x.device)#소스
            source_img_v_s = info_s['feature'][feature_v_name].to(x.device)#소스
        
            mask_indices = info['mask_indices']
            #원래는 이렇게 했음
            #source_img_k_s[:, :, mask_indices, ...] = source_img_k[:, :, mask_indices, ...]
            
            #source_img_v_s[:, :, mask_indices, ...] = source_img_v[:, :, mask_indices, ...] 
            #source_img_k_s[:, :, mask_indices, ...] = img_k
            
            #source_img_v_s[:, :, mask_indices, ...] = img_v

            print(info['t'])
            
            if info['t'] < 0.745 and 15 < info['id'] < 31:
                print("실행!")

                info['mask'] = info['union_mask']
                info['mask_indices'] = info['union_mask_indices']
                source_img_k_s[:, :, mask_indices, ...] = source_img_k[:, :, mask_indices, ...]
                source_img_v_s[:, :, mask_indices, ...] = source_img_v[:, :, mask_indices, ...]
                #source_img_k_s[:, :, mask_indices, ...] = img_k
                #source_img_v_s[:, :, mask_indices, ...] = img_v
            else:
                source_img_k_s[:, :, mask_indices, ...] = img_k
                source_img_v_s[:, :, mask_indices, ...] = img_v
                
            
            
            k = torch.cat((txt_k, source_img_k_s), dim=2)
            v = torch.cat((txt_v, source_img_v_s), dim=2)
            attn, attn_weights = attention(q, k, v, pe=pe, pe_q = info['pe_mask'],attention_mask=info['attention_scale'], return_weights=True)
            txt_len = 512
            img_len = source_img_k_s.shape[2]
            attn_text_to_img = attn_weights[:, :, :txt_len, txt_len:txt_len+img_len]
            if info.get("track_cross", False):
                info["tracker"].add(attn_text_to_img)
                
            # === 여기서 cross-attention map 저장 ===

            # save_cross_attention_map(
            #     attn_weights=attn_weights,
            #     info=info,
            #     txt_len=txt_len,
            #     img_len=img_len,
            #     layer_tag="SB",
            #     inp_target_s=inp_target_s,
            # )
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output
