```
python cli_kv_edit.py \
    --input_image "003.jpg" \
    --mask_image "ref_mask3.jpg" \
    --ref_image "011.png" \
    --ref_mask_image "ref_mask3.jpg" \
    --source_prompt "a bag is on the floor" \
    --target_prompt "a dog is sitting on the floor" \
    --re_init
```

# Subject-driven Image Editing Research

This repository summarizes my undergraduate research experience on **subject-driven image editing** using diffusion-based generative models.

The main focus of this work was to explore how to preserve the identity of a reference subject while naturally editing or inserting it into a target image.

> This repository was originally started as an experimental workspace for KV-Edit-based feature injection experiments.

---

## Research Experience

- **Position:** Undergraduate Research Student
- **Lab:** Multimodal AI Lab (Inha University)
- **Duration:** 1 year and 4 months
- **Research Area:** Subject-driven image editing, diffusion-based generative models, attention / feature-level image editing

---

## Motivation & Research Background

Subject-driven image editing is a promising technology with practical applications in various industries, including virtual try-on, interior design, product placement, and personalized content creation.

I became interested in this field because it can provide real user value while involving challenging research problems such as reference identity preservation, natural object insertion, background consistency, and edit controllability.

At the beginning of my research in early 2025, I first investigated how recent subject-driven image editing and reference-based generation methods performed in practice. I reviewed and tested several related methods, including **AnyDoor**, **Tuning-Free Image Customization with Image and Text Guidance (ECCV 2024)**, **SISO**, **InstantSwap**, and **MasaCtrl**.

Through this process, I observed that many existing methods could edit or insert the target subject to some extent, but they often caused unintended changes in regions that should have remained unchanged. In particular, preserving the surrounding background and layout while modifying only the desired subject region was still a challenging problem.

Based on these observations, I focused my undergraduate research on diffusion-based subject-driven image editing, especially on attention analysis and feature-level manipulation for improving identity preservation and background consistency.


## Research Direction

I selected **KV-Edit(ICCV 2025)** as the base model for this research because I believed it could effectively preserve the surrounding background during image editing.

Background preservation is an important challenge in subject-driven image editing, since the model must edit only the desired region while keeping the rest of the image visually consistent.

However, KV-Edit is fundamentally a generation model, and it cannot directly perform reference-image-based image editing. In other words, it has limitations in understanding a given reference image and using it to preserve the subject's identity during the editing process.

To address this limitation, I conducted research on extending the KV-Edit framework so that it could better understand reference images, preserve subject identity, and generate edited images that also follow the given text condition.

The main goal of this research was to explore how a generative editing model can incorporate reference-image information while maintaining both identity preservation and background consistency.




---

### 1. KV-Edit-based Feature Injection

To enable reference-based editing in the generative KV-Edit model, I investigated feature injection strategies for better identity preservation:

Goal: Enable the model to understand reference subjects while following text conditions.

Method: Conducted initial experiments by injecting reference Key/Value features into all timesteps and attention layers of the generation process.


![KV-Edit Result](./assets/no1_exp_iamge1.jpg)

---

### 3. Attention Map Visualization

I visualized token-level attention maps to analyze whether object-related text tokens were properly localized in the generated image.

This helped inspect failure cases where the model did not correctly attend to the intended object region.

Example visualization:

| Input Image | Attention Map |
|---|---|
| ![](./assets/input_example.png) | ![](./assets/attention_map_example.png) |

---

### 4. Diffusion Feature and Object-level Analysis

I also explored how diffusion model features represent object-level information.

One direction was to use clustering-based analysis on diffusion features to check whether object regions could be separated in a bottom-up manner.

This was considered as a possible way to refine masks or identify object-related regions without relying only on text attention.

---

## Experiment Results

Add your experiment results here.

### Object Editing Example

| Reference Subject | Target Image | Edited Result |
|---|---|---|
| ![](./assets/reference.png) | ![](./assets/target.png) | ![](./assets/result.png) |

### Attention Visualization Example

| Prompt Token | Attention Map |
|---|---|
| `dog` | ![](./assets/dog_attention.png) |

---

## Lab Meeting Feedback and Future Directions

Based on lab meeting discussions, I organized several possible future directions:

- Combining Stable Diffusion features with DINO features
- Using diffusion feature clustering for object region refinement
- Comparing single-image and two-image subject swapping frameworks
- Exploring frequency-domain manipulation for preserving high-frequency details
- Investigating inversion-based editing for real image manipulation
- Selecting effective layers for attention or feature injection
- Designing proper baselines for identity preservation evaluation

---

## Key Takeaways

Through this research experience, I learned how to:

- read and analyze recent generative AI papers
- understand diffusion-based image editing pipelines
- inspect attention and feature representations inside diffusion models
- modify model components for feature-level experiments
- use visualization to debug model behavior
- convert broad research ideas into concrete experiment plans

---

## Tech Stack

- Python
- PyTorch
- Diffusers
- Stable Diffusion
- FLUX / DiT-based architectures
- Attention visualization
- Image editing pipelines

---

## Repository Structure

```text
subject-driven-image-editing/
├── README.md
├── assets/
│   ├── input_example.png
│   ├── attention_map_example.png
│   ├── reference.png
│   ├── target.png
│   └── result.png
├── notes/
│   └── paper_summary.md
└── experiments/
    └── kv_feature_injection/
