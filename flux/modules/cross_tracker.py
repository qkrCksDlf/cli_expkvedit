# flux/modules/cross_tracker.py

import os
import numpy as np
import torch
from PIL import Image


class CrossAttentionTracker:
    """
    특정 토큰(dog 등)에 대한 cross-attention을 t별로 수집하여
    (H,W) mask를 생성하고 저장하는 분석용 트래커.
    """

    def __init__(self, token_idx, out_root="attn_token_masks", attn_id="0"):
        self.token_idx = token_idx
        self.out_root = out_root
        self.attn_id = attn_id

        self.cross_attn_list = []   # 각 layer에서 수집한 attn [B,H,txt,img]

    def add(self, attn):
        """
        attn: [B, H, txt_len, img_len]
        """
        self.cross_attn_list.append(attn.detach().cpu())

    def save_mask(self, t):
        if len(self.cross_attn_list) == 0:
            return

        # (1) 모든 block attn → mean
        # list[L] → [L,B,H,txt,img]
        attn = torch.stack(self.cross_attn_list, dim=0)
        attn = attn.mean(0).mean(1)   # block 평균 + head 평균 → [B, txt, img]

        # (2) dog 토큰 index 선택
        if self.token_idx >= attn.shape[1]:
            print(f"[Tracker] WARNING token_idx {self.token_idx} out of range")
            return

        dog_vec = attn[0, self.token_idx]  # [img_len]

        # (3) reshape H×W 자동 계산
        img_len = dog_vec.shape[0]

        side = int(np.sqrt(img_len))
        if side * side == img_len:
            h = w = side
        else:
            # factorization
            h = int(np.sqrt(img_len))
            while h > 1 and img_len % h != 0:
                h -= 1
            w = img_len // h

        heat = dog_vec.view(h, w)

        # (4) normalize
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-10)

        # (5) save
        t_str = f"{float(t):.3f}".replace(".", "_")
        out_dir = os.path.join(self.out_root, str(self.attn_id), f"t_{t_str}")
        os.makedirs(out_dir, exist_ok=True)

        arr = (heat.numpy() * 255).astype("uint8")
        img = Image.fromarray(arr, mode="L")
        img = img.resize((w * 4, h * 4), Image.NEAREST)
        img.save(os.path.join(out_dir, "mask.png"))

        print(f"✅ Tracker saved mask t={t:.3f} at {out_dir}")

        # 다음 step을 위해 clear
        self.cross_attn_list = []
