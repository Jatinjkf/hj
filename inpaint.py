import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image, ImageOps
import argparse
import requests
from io import BytesIO
import utils

# Suppress warnings
utils.suppress_warnings()

def load_image(image_path):
    try:
        if image_path.startswith("http"):
            response = requests.get(image_path)
            return Image.open(BytesIO(response.content)).convert("RGB")
        else:
            return Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Inpaint an image using Tiny-SD on CPU (Img2Img workaround).")
    parser.add_argument("--image", type=str, required=True, help="Path or URL to the input image.")
    parser.add_argument("--mask", type=str, required=True, help="Path or URL to the mask image (white for area to change).")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt to guide the inpainting.")
    parser.add_argument("--output", type=str, default="output_inpaint.png", help="Output file name.")
    parser.add_argument("--steps", type=int, default=15, help="Number of inference steps (Default: 15).")
    parser.add_argument("--strength", type=float, default=0.75, help="Transformation strength.")
    
    args = parser.parse_args()
    args.output = utils.ensure_extension(args.output)

    model_id = "segmind/tiny-sd"
    
    print(f"Loading model {model_id}...")
    try:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float32, use_safetensors=True)
    except Exception as e:
         print(f"Safetensors load failed, falling back to standard weights...")
         pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float32, use_safetensors=False)

    pipe = pipe.to("cpu")
    pipe.enable_attention_slicing()
    pipe.scheduler = utils.get_optimal_scheduler(pipe)

    print("Loading image and mask...")
    init_image = load_image(args.image)
    mask_image = load_image(args.mask)
    if not init_image or not mask_image: return

    init_image = init_image.resize((512, 512))
    mask_image = mask_image.convert("L").resize((512, 512))

    print(f"Processing inpainting via Img2Img...")
    generated_image = pipe(
        prompt=args.prompt, 
        image=init_image, 
        strength=args.strength, 
        num_inference_steps=args.steps
    ).images[0]

    final_image = Image.composite(generated_image, init_image, mask_image)

    final_image.save(args.output)
    print(f"Image saved to {args.output}")

if __name__ == "__main__":
    main()
