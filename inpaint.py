import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image, ImageOps
import argparse
import requests
from io import BytesIO

def load_image(image_path):
    if image_path.startswith("http"):
        response = requests.get(image_path)
        return Image.open(BytesIO(response.content)).convert("RGB")
    else:
        return Image.open(image_path).convert("RGB")

def main():
    parser = argparse.ArgumentParser(description="Inpaint an image using Tiny-SD on CPU (Img2Img workaround).")
    parser.add_argument("--image", type=str, required=True, help="Path or URL to the input image.")
    parser.add_argument("--mask", type=str, required=True, help="Path or URL to the mask image (white for area to change).")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt to guide the inpainting.")
    parser.add_argument("--output", type=str, default="output_inpaint.png", help="Output file name.")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps.")
    parser.add_argument("--strength", type=float, default=0.75, help="Transformation strength.")
    
    args = parser.parse_args()

    model_id = "segmind/tiny-sd"
    
    print(f"Loading model {model_id}...")
    # Use Img2Img pipeline because Tiny-SD doesn't have a 9-channel inpainting UNet
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe = pipe.to("cpu")
    pipe.enable_attention_slicing()

    print("Loading image and mask...")
    init_image = load_image(args.image).resize((512, 512))
    mask_image = load_image(args.mask).convert("L").resize((512, 512))

    print(f"Processing inpainting via Img2Img...")
    # For a 4-channel model, we use Img2Img and then blend
    generated_image = pipe(
        prompt=args.prompt, 
        image=init_image, 
        strength=args.strength, 
        num_inference_steps=args.steps
    ).images[0]

    # Blend original and generated image using the mask
    # mask_image: white (255) is the area we want to keep from generated_image
    final_image = Image.composite(generated_image, init_image, mask_image)

    final_image.save(args.output)
    print(f"Image saved to {args.output}")

if __name__ == "__main__":
    main()
