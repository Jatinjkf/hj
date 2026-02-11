# Tiny-SD CPU Studio

This repository contains a collection of Python scripts to run the **Segmind Tiny-SD** model on CPU. Tiny-SD is a distilled, lightweight version of Stable Diffusion 1.5, making it ideal for CPU-based generation.

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## The Easy Way: Tiny-SD Studio
Run the main interactive script for a cool, user-friendly terminal interface:
```bash
python main.py
```

### Pro Version (CPU & Intel GPU Support)
If you have an **Intel Integrated GPU (like HD 620)**, use the Pro version to choose between CPU and GPU acceleration:
```bash
python main_pro.py
```
*Note: To use your Intel GPU on Windows, it is highly recommended to install `torch-directml` (`pip install torch-directml`).*

---

## Individual Scripts
If you prefer running specific tasks via command line arguments, you can use these individual scripts:

### 1. Text-to-Image
Generate an image from a text prompt.
```bash
python txt2img.py --prompt "A futuristic city in the clouds" --output city.png
```

### 2. Image-to-Image
Modify an existing image using a prompt.
```bash
python img2img.py --image input.jpg --prompt "Make it a Van Gogh painting" --strength 0.6
```

### 3. Inpainting
Edit a specific part of an image using a mask.
```bash
python inpaint.py --image base.png --mask mask.png --prompt "A cat sitting on the chair"
```

### 4. Outpainting
Extend the borders of an image.
```bash
python outpaint.py --image original.png --prompt "A wide landscape with mountains" --padding 128
```

### 5. Image Variations
Generate multiple similar versions of an input image.
```bash
python variations.py --image input.png --num_variations 3 --strength 0.4
```

### 6. LoRA Support
Generate an image using specific LoRA weights.
```bash
python lora_txt2img.py --prompt "A portrait in the style of <lora-name>" --lora "path/to/lora_weights.safetensors"
```

## Performance Optimizations
-   **CPU-Friendly:** All scripts use `float32` and `pipe.enable_attention_slicing()` to minimize RAM usage and optimize CPU performance.
-   **Speed:** Tiny-SD is up to 80% faster than standard SD 1.5. Expect generations to take ~30-60 seconds on a typical modern CPU.
