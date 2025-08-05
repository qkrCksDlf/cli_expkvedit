import os

import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
import os
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

from diffusers.models.attention_processor import (
    AttnProcessor,
    AttnProcessor2_0,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    JointAttnProcessor2_0,
    FluxAttnProcessor2_0
)

from .modules import *


def hook_function(name, detach=True):
    def forward_hook(module, input, output):
        if hasattr(module.processor, "attn_map"):

            timestep = module.processor.timestep

            attn_maps[timestep] = attn_maps.get(timestep, dict())
            attn_maps[timestep][name] = module.processor.attn_map.cpu() if detach \
                else module.processor.attn_map
            
            del module.processor.attn_map

    return forward_hook


def register_cross_attention_hook(model, hook_function, target_name):
    for name, module in model.named_modules():
        if not name.endswith(target_name):
            continue

        if isinstance(module.processor, AttnProcessor):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, AttnProcessor2_0):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, LoRAAttnProcessor):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, LoRAAttnProcessor2_0):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, JointAttnProcessor2_0):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, FluxAttnProcessor2_0):
            module.processor.store_attn_map = True

        hook = module.register_forward_hook(hook_function(name))
    
    return model


def replace_call_method_for_unet(model):
    if model.__class__.__name__ == 'UNet2DConditionModel':
        from diffusers.models.unets import UNet2DConditionModel
        model.forward = UNet2DConditionModelForward.__get__(model, UNet2DConditionModel)

    for name, layer in model.named_children():
        
        if layer.__class__.__name__ == 'Transformer2DModel':
            from diffusers.models import Transformer2DModel
            layer.forward = Transformer2DModelForward.__get__(layer, Transformer2DModel)
        
        elif layer.__class__.__name__ == 'BasicTransformerBlock':
            from diffusers.models.attention import BasicTransformerBlock
            layer.forward = BasicTransformerBlockForward.__get__(layer, BasicTransformerBlock)
        
        replace_call_method_for_unet(layer)
    
    return model


# TODO: implement
# def replace_call_method_for_sana(model):
#     if model.__class__.__name__ == 'SanaTransformer2DModel':
#         from diffusers.models.transformers import SanaTransformer2DModel
#         model.forward = SanaTransformer2DModelForward.__get__(model, SanaTransformer2DModel)

#     for name, layer in model.named_children():
        
#         if layer.__class__.__name__ == 'SanaTransformerBlock':
#             from diffusers.models.transformers.sana_transformer import SanaTransformerBlock
#             layer.forward = SanaTransformerBlockForward.__get__(layer, SanaTransformerBlock)
        
#         replace_call_method_for_sana(layer)
    
#     return model


def replace_call_method_for_sd3(model):
    if model.__class__.__name__ == 'SD3Transformer2DModel':
        from diffusers.models.transformers import SD3Transformer2DModel
        model.forward = SD3Transformer2DModelForward.__get__(model, SD3Transformer2DModel)

    for name, layer in model.named_children():
        
        if layer.__class__.__name__ == 'JointTransformerBlock':
            from diffusers.models.attention import JointTransformerBlock
            layer.forward = JointTransformerBlockForward.__get__(layer, JointTransformerBlock)
        
        replace_call_method_for_sd3(layer)
    
    return model


def replace_call_method_for_flux(model):
    if model.__class__.__name__ == 'FluxTransformer2DModel':
        from diffusers.models.transformers import FluxTransformer2DModel
        model.forward = FluxTransformer2DModelForward.__get__(model, FluxTransformer2DModel)

    for name, layer in model.named_children():
        
        if layer.__class__.__name__ == 'FluxTransformerBlock':
            from diffusers.models.transformers.transformer_flux import FluxTransformerBlock
            layer.forward = FluxTransformerBlockForward.__get__(layer, FluxTransformerBlock)
        
        replace_call_method_for_flux(layer)
    
    return model


def init_pipeline(pipeline):
    AttnProcessor.__call__ = attn_call
    AttnProcessor2_0.__call__ = attn_call2_0
    LoRAAttnProcessor.__call__ = lora_attn_call
    LoRAAttnProcessor2_0.__call__ = lora_attn_call2_0
    if 'transformer' in vars(pipeline).keys():
        if pipeline.transformer.__class__.__name__ == 'SD3Transformer2DModel':
            JointAttnProcessor2_0.__call__ = joint_attn_call2_0
            pipeline.transformer = register_cross_attention_hook(pipeline.transformer, hook_function, 'attn')
            pipeline.transformer = replace_call_method_for_sd3(pipeline.transformer)
        
        elif pipeline.transformer.__class__.__name__ == 'FluxTransformer2DModel':
            from diffusers import FluxPipeline
            FluxAttnProcessor2_0.__call__ = flux_attn_call2_0
            FluxPipeline.__call__ = FluxPipeline_call
            pipeline.transformer = register_cross_attention_hook(pipeline.transformer, hook_function, 'attn')
            pipeline.transformer = replace_call_method_for_flux(pipeline.transformer)

        # TODO: implement
        # elif pipeline.transformer.__class__.__name__ == 'SanaTransformer2DModel':
        #     from diffusers import SanaPipeline
        #     SanaPipeline.__call__ == SanaPipeline_call
        #     pipeline.transformer = register_cross_attention_hook(pipeline.transformer, hook_function, 'attn2')
        #     pipeline.transformer = replace_call_method_for_sana(pipeline.transformer)

    else:
        if pipeline.unet.__class__.__name__ == 'UNet2DConditionModel':
            pipeline.unet = register_cross_attention_hook(pipeline.unet, hook_function, 'attn2')
            pipeline.unet = replace_call_method_for_unet(pipeline.unet)


    return pipeline


def process_token(token, startofword):
    if '</w>' in token:
        token = token.replace('</w>', '')
        if startofword:
            token = '<' + token + '>'
        else:
            token = '-' + token + '>'
            startofword = True
    elif token not in ['<|startoftext|>', '<|endoftext|>']:
        if startofword:
            token = '<' + token + '-'
            startofword = False
        else:
            token = '-' + token + '-'
    return token, startofword


def save_attention_image(attn_map, tokens, batch_dir, to_pil):
    startofword = True
    for i, (token, a) in enumerate(zip(tokens, attn_map[:len(tokens)])):
        token, startofword = process_token(token, startofword)
        to_pil(a.to(torch.float32)).save(os.path.join(batch_dir, f'{i}-{token}.png'))


def save_attention_maps(attn_maps, tokenizer, prompts, base_dir='attn_maps', unconditional=True, hw=(48, 32)):
    """
    Save attention maps (token-level) as heatmaps.

    Args:
        attn_maps: Dict[timestep][layer] → Tensor [B, H, Q, K] or [B, Q, K]
        tokenizer: Tokenizer object (e.g., CLIPTokenizer)
        prompts: List of prompt strings
        base_dir: Where to save
        unconditional: If True, skip unconditional token handling
        hw: Tuple[int, int] for reshaping attention vectors (e.g., (48, 32) → 1536 = K)
    """



    h, w = hw
    os.makedirs(base_dir, exist_ok=True)

    # Tokenize and get text tokens
    tokenized = tokenizer(prompts, padding='max_length', truncation=True, max_length=77, return_tensors="pt")
    input_ids = tokenized["input_ids"]
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()
    total_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]

    total_attn_map = None
    total_attn_map_number = 0

    for timestep, layers in attn_maps.items():
        timestep_dir = os.path.join(base_dir, f'{timestep}')
        os.makedirs(timestep_dir, exist_ok=True)

        for layer, attn_map in layers.items():
            layer_dir = os.path.join(timestep_dir, f'{layer}')
            os.makedirs(layer_dir, exist_ok=True)

            # Normalize attention map shape
            if attn_map.dim() == 4:
                attn_map = attn_map.sum(1)  # [B, Q, K]
            elif attn_map.dim() != 3:
                raise ValueError(f"Unexpected attention map shape: {attn_map.shape}")

            B, Q, K = attn_map.shape
            if K != h * w:
                print(f"[Warning] K={K} does not match expected hw=({h}*{w}={h*w}). Skipping...")
                continue

            attn_map = attn_map.unsqueeze(1)  # (B, 1, Q, K)

            if total_attn_map is None:
                total_attn_map = torch.zeros_like(attn_map)

            resized_attn_map = F.interpolate(attn_map, size=attn_map.shape[-2:], mode='bilinear', align_corners=False)
            total_attn_map += resized_attn_map
            total_attn_map_number += 1

            for batch_idx, (token_list, attn_2d) in enumerate(zip(total_tokens, attn_map)):
                batch_dir = os.path.join(layer_dir, f'batch-{batch_idx}')
                os.makedirs(batch_dir, exist_ok=True)

                attn_2d = attn_2d.squeeze(0)  # (Q, K)

                for token_idx, (token, attn_vec) in enumerate(zip(token_list, attn_2d)):
                    token = token.replace("</w>", "").strip()
                    attn_np = attn_vec.detach().cpu().float().numpy()

                    if attn_np.shape[0] != h * w:
                        print(f"[Skip] Token {token} has mismatched shape: {attn_np.shape}")
                        continue

                    attn_2d_map = attn_np.reshape(h, w)
                    attn_2d_map = (attn_2d_map - attn_2d_map.min()) / (attn_2d_map.max() - attn_2d_map.min() + 1e-8)
                    heatmap = cv2.applyColorMap(np.uint8(255 * attn_2d_map), cv2.COLORMAP_JET)
                    Image.fromarray(heatmap).save(os.path.join(batch_dir, f"{token_idx}-{token}.png"))

    # Save averaged attention map
    if total_attn_map_number > 0:
        total_attn_map /= total_attn_map_number

        for batch_idx, (attn_map, tokens) in enumerate(zip(total_attn_map, total_tokens)):
            batch_dir = os.path.join(base_dir, f'batch-{batch_idx}')
            os.makedirs(batch_dir, exist_ok=True)
            attn_map = attn_map.squeeze(0)  # (Q, K)

            for token_idx, (token, attn_vec) in enumerate(zip(tokens, attn_map)):
                token = token.replace("</w>", "").strip()
                attn_np = attn_vec.detach().cpu().float().numpy()

                if attn_np.shape[0] != h * w:
                    continue
                attn_2d_map = attn_np.reshape(h, w)
                attn_2d_map = (attn_2d_map - attn_2d_map.min()) / (attn_2d_map.max() - attn_2d_map.min() + 1e-8)
                heatmap = cv2.applyColorMap(np.uint8(255 * attn_2d_map), cv2.COLORMAP_JET)
                Image.fromarray(heatmap).save(os.path.join(batch_dir, f"{token_idx}-{token}.png"))

    print(f"✅ All attention maps saved in: {base_dir}")

