# KV-Edit CLI Demo

This directory contains a command-line interface (CLI) version of the KV-Edit image editing system, converted from the original Gradio web interface.

## 🚀 Quick Start

### 1. Setup
```bash
# Option A: Automatic setup (recommended)
./setup_cli.sh

# Option B: Manual setup
# Activate conda environment
conda activate KV-Edit

# Make scripts executable
chmod +x *.sh

# Install dependencies (if not already done)
pip install -r requirements.txt
```

### 2. Prepare Your Images
You need two images:
- **Input image**: The image you want to edit
- **Mask image**: A grayscale/binary mask where:
  - **White areas** = regions to be edited
  - **Black areas** = regions to preserve unchanged

### 3. Create a sample mask (optional)
```bash
# Activate environment first
conda activate KV-Edit

# Create sample masks for testing
python create_sample_mask.py your_image.jpg mask_center.png center
python create_sample_mask.py your_image.jpg mask_left.png left
```

### 4. Run Basic Editing
```bash
# Simple usage (conda environment will be activated automatically)
./quick_start.sh your_image.jpg your_mask.png "original description" "desired result"

# Example
./quick_start.sh photo.jpg mask.png "a person in a garden" "a robot in a garden"
```

## 📋 Available Scripts

### `setup_cli.sh` - Complete Environment Setup
One-time setup script that handles everything:
```bash
./setup_cli.sh
```
This script will:
- Check conda installation
- Create 'KV-Edit' conda environment
- Install PyTorch with CUDA support
- Install all dependencies
- Setup execution scripts
- Create helper utilities

### `quick_start.sh` - Simple Usage
For quick testing and basic usage:
```bash
./quick_start.sh <input> <mask> <source_prompt> <target_prompt> [extra_options]
```

### `run_cli_example.sh` - Basic Example
Template script with predefined settings. Edit the variables at the top:
```bash
# Edit the script to set your paths
vim run_cli_example.sh

# Then run
./run_cli_example.sh
```

### `run_cli_advanced.sh` - Advanced Examples
Interactive script with multiple preset configurations:
```bash
# Interactive mode
./run_cli_advanced.sh

# Or run specific examples directly
./run_cli_advanced.sh basic      # Basic editing
./run_cli_advanced.sh quality    # High quality (slow)
./run_cli_advanced.sh fast       # Fast preview
./run_cli_advanced.sh precise    # Precise background preservation
./run_cli_advanced.sh creative   # Creative editing
./run_cli_advanced.sh resolution # Custom resolution
```

## 🎛️ Direct CLI Usage

### Manual Environment Activation
If you need to activate the environment manually:
```bash
conda activate KV-Edit
```

### Basic Command
```bash
python cli_kv_edit.py \
    --input_image path/to/image.jpg \
    --mask_image path/to/mask.png \
    --source_prompt "description of original image" \
    --target_prompt "description of desired result"
```

### Advanced Command with All Options
```bash
python cli_kv_edit.py \
    --input_image photo.jpg \
    --mask_image mask.png \
    --source_prompt "a person standing in a garden" \
    --target_prompt "a robot standing in a garden" \
    --output_dir my_results \
    --inversion_num_steps 28 \
    --denoise_num_steps 28 \
    --skip_step 4 \
    --inversion_guidance 1.5 \
    --denoise_guidance 5.5 \
    --attn_scale 1.0 \
    --seed 42 \
    --re_init \
    --attn_mask \
    --name flux-dev \
    --device cuda \
    --gpus
```

## ⚙️ Parameter Guide

### Required Parameters
- `--input_image`: Path to your input image
- `--mask_image`: Path to mask image (white=edit, black=preserve)
- `--source_prompt`: Description of the original image
- `--target_prompt`: Description of the desired result

### Quality & Performance
- `--inversion_num_steps` (default: 28): DDIM inversion steps
  - Higher = more accurate inversion but slower
- `--denoise_num_steps` (default: 28): Denoising steps  
  - Higher = better quality but slower
- `--skip_step` (default: 4): Steps to skip during editing
  - Lower = more faithful to prompt, may affect background

### Guidance Controls
- `--inversion_guidance` (default: 1.5): Inversion guidance scale
  - Higher = stronger adherence to source prompt
- `--denoise_guidance` (default: 5.5): Denoising guidance scale
  - Higher = stronger adherence to target prompt
- `--attn_scale` (default: 1.0): Attention scale for background preservation
  - Higher = better background preservation

### Advanced Options
- `--re_init`: Enable re-initialization (better editing, may affect background)
- `--attn_mask`: Enable attention masking (enhanced performance)
- `--seed` (default: 42): Random seed (-1 for random)

### System Options
- `--name` (default: flux-dev): Model to use (flux-dev or flux-schnell)
- `--device` (default: cuda): Device (cuda or cpu)
- `--gpus`: Use two GPUs for distributed processing
- `--output_dir` (default: regress_result): Output directory
- `--width/--height`: Custom dimensions (auto-detected from image)

## 🎯 Usage Examples

### Basic Editing
```bash
python cli_kv_edit.py \
    --input_image photo.jpg \
    --mask_image mask.png \
    --source_prompt "a cat sitting on a chair" \
    --target_prompt "a dog sitting on a chair"
```

### High Quality Editing (Slower)
```bash
python cli_kv_edit.py \
    --input_image photo.jpg \
    --mask_image mask.png \
    --source_prompt "a person in a room" \
    --target_prompt "a robot in a room" \
    --inversion_num_steps 50 \
    --denoise_num_steps 50 \
    --skip_step 2 \
    --denoise_guidance 7.5 \
    --attn_scale 1.5
```

### Fast Preview (Lower Quality)
```bash
python cli_kv_edit.py \
    --input_image photo.jpg \
    --mask_image mask.png \
    --source_prompt "a car on a road" \
    --target_prompt "a truck on a road" \
    --inversion_num_steps 15 \
    --denoise_num_steps 15 \
    --skip_step 6 \
    --name flux-schnell
```

### Precise Background Preservation
```bash
python cli_kv_edit.py \
    --input_image photo.jpg \
    --mask_image mask.png \
    --source_prompt "a person in a landscape" \
    --target_prompt "a statue in a landscape" \
    --skip_step 1 \
    --attn_scale 2.0 \
    --attn_mask \
    --re_init \
    --inversion_guidance 2.0
```

## 📁 Output Structure

Results are saved to the specified output directory (default: `regress_result/`):
```
regress_result/
├── img_0.jpg       # Edited image
├── img_0_mask.png  # Copy of the mask used
├── img_1.jpg       # Next edit (if multiple runs)
└── img_1_mask.png
```

## 🔧 Creating Mask Images

You can create mask images using any image editing software:

### Using the included helper script
```bash
# Activate environment
conda activate KV-Edit

# Create different types of masks
python create_sample_mask.py input.jpg mask_center.png center
python create_sample_mask.py input.jpg mask_left.png left
python create_sample_mask.py input.jpg mask_right.png right
python create_sample_mask.py input.jpg mask_top.png top
python create_sample_mask.py input.jpg mask_bottom.png bottom
```

### Using GIMP/Photoshop
1. Open your input image
2. Create a new layer
3. Paint white on areas you want to edit
4. Paint black on areas you want to preserve
5. Export as PNG

### Using Python (programmatic)
```python
from PIL import Image, ImageDraw
import numpy as np

# Create a mask
img = Image.open('input.jpg')
mask = Image.new('L', img.size, 0)  # Black background
draw = ImageDraw.Draw(mask)

# Draw white rectangle for edit area
draw.rectangle([100, 100, 300, 300], fill=255)
mask.save('mask.png')
```

### Using ImageMagick (command line)
```bash
# Create circular mask
convert input.jpg -fill black -colorize 100 \
        -fill white -draw "circle 200,200 200,100" \
        mask.png
```

## ⚡ Performance Tips

### For Better Speed
- Use `flux-schnell` model (`--name flux-schnell`)
- Reduce steps (`--inversion_num_steps 15 --denoise_num_steps 15`)
- Increase skip steps (`--skip_step 6`)

### For Better Quality  
- Use `flux-dev` model (default)
- Increase steps (`--inversion_num_steps 50 --denoise_num_steps 50`)
- Reduce skip steps (`--skip_step 2`)
- Add quality flags (`--attn_mask --re_init`)

### For Better Background Preservation
- Lower skip steps (`--skip_step 1`)
- Higher attention scale (`--attn_scale 2.0`)
- Enable attention masking (`--attn_mask`)
- Higher inversion guidance (`--inversion_guidance 2.0`)

## 🐛 Troubleshooting

### Environment Issues

**"conda: command not found"**
- Install Miniconda or Anaconda
- Add conda to your PATH
- Restart your terminal

**"Environment 'KV-Edit' not found"**
- Run the setup script: `./setup_cli.sh`
- Or create manually: `conda create -n KV-Edit python=3.9`

**"Failed to activate conda environment"**
- Check environment exists: `conda env list`
- Try manual activation: `conda activate KV-Edit`
- Ensure conda is properly initialized

### Common Issues

**"CUDA out of memory"**
- Reduce image size or use `--device cpu`
- Use flux-schnell model
- Use single GPU (remove `--gpus` flag)

**"Module not found"**
- Activate environment: `conda activate KV-Edit`
- Install requirements: `pip install -r requirements.txt`
- Run setup script: `./setup_cli.sh`

**"Image dimensions not multiple of 16"**
- The CLI automatically adjusts dimensions
- Or specify custom dimensions: `--width 1024 --height 1024`

**Poor editing quality**
- Try different skip_step values (2-6)
- Adjust guidance values
- Enable attention flags (`--attn_mask --re_init`)
- Use higher quality settings

**Background not preserved**
- Lower skip_step (1-2)
- Increase attn_scale (1.5-2.0)
- Enable `--attn_mask`
- Check mask quality (should be clean white/black)

## 📚 Additional Resources

- Original Gradio demo: `gradio_kv_edit_gpu.py`
- Model documentation: Check `flux/` directory
- Paper and project page: [KV-Edit GitHub](https://github.com/Xilluill/KV-Edit)

## 🤝 Support

For issues and questions:
1. Check this README for common solutions
2. Review parameter explanations
3. Try different parameter combinations
4. Check the original repository for updates 