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
            #source_img_k[:, :, mask_indices, ...] = img_k
            #source_img_v[:, :, mask_indices, ...] = img_v
            
            source_img_k_s[:, :, mask_indices, ...] = img_k_r[:, :, mask_indices, :]
            source_img_v_s[:, :, mask_indices, ...] = img_v_r[:, :, mask_indices, :]

            
            
            
            
            
            q = torch.cat((txt_q, img_q), dim=2) #소스이미지
            k = torch.cat((txt_k, source_img_k_s), dim=2) 
            v = torch.cat((txt_v, source_img_v_s), dim=2)



            #attn = attention(q, k, v, pe=pe, pe_q = info['pe_mask'],attention_mask=info['attention_scale'])
            attn, attn_weights = attention(q, k, v, pe=pe, pe_q=info['pe_mask'], attention_mask=info['attention_scale'], return_weights=True)

            '''
            =================================================================================================================================
            text and image attention 관련 
            '''
            #layer_name = f"{info['id']}_{info['t']}_DoubleStreamBlock_kv"
            #timestep = 0  # 혹시 모를 다중 저장 대비
            
            # 1. attention weights 저장
            #attn_maps[timestep] = attn_maps.get(timestep, dict())
            #attn_maps[timestep][layer_name] = attn_weights.detach().cpu()
            # save_attention_maps(
            #     attn_maps,
            #     tokenizer=info['tokenizer'],         # info에 tokenizer 추가되어야 함
            #     prompts="a dog is lying on the floor",           # info에 caption도 있어야 함
            #     base_dir='attn_overlay',
            #     token_filter='dog',
            #     hw=(48, 32)
            # )
            '''
            =================================================================================================================================
            '''

            
            overlay_attention_map(
                str(info['id']),
                str(info['t']),
                attn_weights=attn_weights,
                q_idx=txt.shape[1] + info['mask_indices'][0],  # 강아지 위치
                h_idx=0,
                img_size=(48,32),
                base_image_path="x1.jpg",
                save_path="attn_overlay/vis1.png"
)
            

        
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
            #추가
            x_mod_r = (1 + mod.scale) * self.pre_norm(zt_r) + mod.shift
            qkv_r, mlp_r = torch.split(self.linear1(x_mod_r), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
    
            q_r, k_r, v_r = rearrange(qkv_r, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
            q_r, k_r = self.norm(q_r, k_r, v_r)
            img_k_r = k_r[:, :, 512:, ...]
            img_v_r = v_r[:, :, 512:, ...]
            ##################################################
                
            source_img_k = info['feature'][feature_k_name].to(x.device) #레퍼런스
            source_img_v = info['feature'][feature_v_name].to(x.device) #레퍼런스

            source_img_k_s = info_s['feature'][feature_k_name].to(x.device)#소스
            source_img_v_s = info_s['feature'][feature_v_name].to(x.device)#소스
        
            mask_indices = info['mask_indices']
            source_img_k_s[:, :, mask_indices, ...] = source_img_k[:, :, mask_indices, ...]
            source_img_v_s[:, :, mask_indices, ...] = source_img_v[:, :, mask_indices, ...]
            
            k = torch.cat((txt_k, source_img_k_s), dim=2)
            v = torch.cat((txt_v, source_img_v_s), dim=2)
            attn = attention(q, k, v, pe=pe, pe_q = info['pe_mask'],attention_mask=info['attention_scale'])

        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output
