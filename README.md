```
python cli_kv_edit.py \
    --input_image "003.jpg" \
    --mask_image "ref_mask3.jpg" \
    --ref_image "011.png" \
    --ref_mask_image "ref_mask2.jpg" \
    --source_prompt "a bag is on the floor" \
    --target_prompt "a dog is sitting on the floor" \
    --attn_mask \
    ----denoise_num_steps 35
```


```
python cli_kv_edit.py \
    --input_image "x1.jpg" \
    --mask_image "x4.jpg" \
    --ref_image "x3.png" \
    --ref_mask_image "x4.jpg" \
    --source_prompt "a cat is lying on the floor" \
    --target_prompt "a dog is lying on the floor" \
    --attn_mask
```

```
python cli_kv_edit.py \
    --input_image "simg_00.jpg" \
    --mask_image "simg_00_mask.png" \
    --ref_image "k.jpg" \
    --ref_mask_image "k_mask.jpg" \
    --source_prompt "a cat is on a pink bed" \
    --target_prompt "a husky toy on a pink bed " \
    --attn_mask \
    --re_init
```
