# Tiny-SD Studio Pro

An optimized Stable Diffusion CLI for Intel CPUs and Integrated GPUs (HD 620, Iris Xe, etc.) using OpenVINO.

## Features
-   **Hardware Acceleration**: Support for OpenVINO (CPU/GPU) and DirectML.
-   **Model Management**: Download models from Hugging Face or URL directly within the app.
-   **Optimization**:
    -   Uses `DPMSolverMultistepScheduler` (DPM++ 2M Karras) for fast generation (15 steps).
    -   Static reshaping and compilation for OpenVINO.
    -   Local model caching to avoid re-downloads/re-conversions.
-   **Advanced Features**:
    -   Text-to-Image, Image-to-Image, Inpainting, Outpainting.
    -   LoRA support (Downloader & Selector included).
    -   Image Variations.
    -   Negative Prompts, Seed control, Batching.

## Capabilities & Compatibility

### Supported Models
This tool is optimized for **Stable Diffusion 1.5** based models.
-   **Recommended**: `segmind/tiny-sd` (Fast, distilled).
-   **Compatible**: `runwayml/stable-diffusion-v1-5`, `SG161222/Realistic_Vision_V5.1_noVAE`, `dreamlike-art/dreamlike-photoreal-2.0`, etc.
-   **Note**: SDXL models are **not** officially supported due to high RAM/VRAM usage and static shape constraints on iGPUs, but may work if you have sufficient hardware (32GB+ RAM).

### Supported LoRAs
-   Any LoRA trained on **SD 1.5** base models (`.safetensors` format recommended).
-   You can download LoRAs directly from Hugging Face or via direct URL (e.g., Civitai) using the built-in downloader.

## Installation

1.  **Install Python 3.10+**
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the main studio script:
```bash
python main_pro.py
```

### 1. Hardware Selection
Choose your accelerator:
-   **OpenVINO GPU**: Best for Intel Integrated Graphics (Iris Xe, HD 620).
-   **OpenVINO CPU**: Fast CPU inference.
-   **Standard CPU**: Slow but compatible fallback.

### 2. Model Selection
The app will ask you to select a model.
-   **Default**: `segmind/tiny-sd` (Fast, distilled SD 1.5).
-   **Download New**: Select this to download any SD 1.5 based model from Hugging Face or Civitai (direct URL).
    -   Downloaded models are saved to `tiny-sd-models/`.
    -   They are automatically converted to OpenVINO format upon first use.

### 3. Generation
Select a task (e.g., Text to Image) and follow the prompts.
-   **Samplers**: Choose from DPM++, Euler, etc.
-   **Dimensions**: For Image-to-Image, the output size matches the input image (rounded to nearest 64px).

### 4. LoRA Usage
-   Select "Download LoRA" from the main menu to get new styles.
-   Select "LoRA Text-to-Image" to use them. You can select a downloaded LoRA from the list.

## Adding Models Manually
You can manually place `.safetensors` files or Diffusers folders into:
-   Models: `tiny-sd-models/`
-   LoRAs: `tiny-sd-models/loras/`

## Troubleshooting
-   **"OpenVINO Load Error"**: If you see errors about `torch.load` vulnerability, the app will try a fallback method. If that fails, consider upgrading `torch`.
-   **Re-conversion**: Models are converted once per resolution/batch-size configuration. Minor reshaping is fast; full conversion happens only once.
