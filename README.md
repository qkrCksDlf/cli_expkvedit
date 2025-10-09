```
python cli_kv_edit.py \
    --input_image "003.jpg" \
    --mask_image "ref_mask2.jpg" \
    --ref_image "011.png" \
    --ref_mask_image "ref_mask2.jpg" \
    --source_prompt "a bag is on the floor" \
    --target_prompt "a dog is sitting on the floor" \
    --attn_mask \
    --re_init
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

```
python demo_lore.py --resize -1 \
                    --source_prompt "a cat is lying on the floor" \
                    --target_prompt "a dog is lying on the floor" \
                    --target_object "dog" \
                    --target_index 5 \
                    --ref_img_dir '../../../cli_expkvedit/x3.png'  \
                    --source_img_dir '../../../cli_expkvedit/x1.jpg' \
                    --source_mask_dir '../../../cli_expkvedit/x4.jpg'  \
                    --seeds 3 \
                    --training_steps 0 \
                    --savename 'floor' \
                    --v_inject 2
```

```
python demo_lore.py --resize -1 \
                    --source_prompt "a bag is on the floor" \
                    --target_prompt "a dog is on the floor" \
                    --target_object "dog" \
                    --target_index 5 \
                    --source_img_dir '../../cli_expkvedit/003.jpg' \
                    --source_mask_dir '../../cli_expkvedit/ref_mask3.jpg'  \
                    --seeds 3 \
                    --savename 'floor'
```

```
python demo_lore.py --resize -1 \
                    --source_prompt "a cat is lying on the floor" \
                    --target_prompt "a dog is lying on the floor" \
                    --target_object "dog" \
                    --target_index 2 \
                    --source_img_dir '../../cli_expkvedit/x1.jpg' \
                    --source_mask_dir '../../cli_expkvedit/x4.jpg'  \
                    --num_steps 10 \
                    --inject 12 \
                    --noise_scale 1 \
                    --training_epochs 1 \
                    --seeds 0 \
                    --savename 'lying'
```


```
python cli_kv_edit.py \
    --input_image "../../cli_expkvedit/x1.jpg" \
    --mask_image "../../cli_expkvedit/x4.jpg" \
    --source_prompt "a cat is lying on the floor" \
    --target_prompt "a dog is lying on the floor" \
    --denoise_guidance 1.5
```
