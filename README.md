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

## Motivation

Subject-driven image editing is a promising technology with practical applications in various industries, including virtual try-on, interior design, product placement, and personalized content creation.

I became interested in this field because it can provide real user value while involving challenging research problems such as identity preservation, natural object insertion, and background consistency.

Based on this motivation, I conducted undergraduate research on diffusion-based subject-driven image editing, focusing on attention analysis and feature-level manipulation.

---


## Overview

Subject-driven image editing aims to modify an image using a reference subject while maintaining both the subject identity and the visual consistency of the target image.

This task is challenging because the model needs to preserve:

- the identity of the reference subject
- the background and layout of the target image
- spatial consistency between the edited region and the original image
- realistic image quality after generation

In this research, I studied diffusion-based editing methods and explored attention / feature-level manipulation strategies for identity-preserving image editing.

---


## Research Direction

I selected **KV-Edit(ICCV 2025)** as the base model for this research because I believed it could effectively preserve the surrounding background during image editing.

Background preservation is an important challenge in subject-driven image editing, since the model must edit only the desired region while keeping the rest of the image visually consistent.

However, KV-Edit is fundamentally a generation model, and it cannot directly perform reference-image-based image editing. In other words, it has limitations in understanding a given reference image and using it to preserve the subject's identity during the editing process.

To address this limitation, I conducted research on extending the KV-Edit framework so that it could better understand reference images, preserve subject identity, and generate edited images that also follow the given text condition.

The main goal of this research was to explore how a generative editing model can incorporate reference-image information while maintaining both identity preservation and background consistency.






## Papers and Methods Reviewed

I studied and analyzed several recent methods related to subject-driven image editing and reference-based generation.

- AnyDoor
- SISO
- KV-Edit
- Stable Flow
- CannyEdit
- Teleportraits
- FLUX / DiT-based editing approaches

The main goal of the paper review was to understand how each method handles reference conditioning, identity preservation, background consistency, and editing controllability.

---

## My Work

### 1. Paper Review and Method Analysis

I reviewed recent papers on subject-driven image editing and compared their core approaches.

The analysis focused on:

- how reference subject information is injected
- how masks are used for local editing
- how attention maps affect object localization
- how inversion is used for image reconstruction and editing
- how identity preservation and background consistency are balanced

---

### 2. KV-Edit-based Feature Injection Exploration

I explored KV-Edit-based feature injection for identity-preserving image editing.

The main idea was to reuse Key / Value features extracted from a reference subject and inject them into selected attention layers or regions during the editing process.

This experiment was motivated by the question:

> Can reference object identity be better preserved by injecting reference-side attention features into the target editing region?

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
