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
    parser = argparse.ArgumentParser(description="Outpaint (extend) an image using Tiny-SD on CPU.")
    parser.add_argument("--image", type=str, required=True, help="Path or URL to the input image.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt describing the whole scene.")
    parser.add_argument("--output", type=str, default="output_outpaint.png", help="Output file name.")
    parser.add_argument("--padding", type=int, default=128, help="Pixels to add to each side.")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps.")
    parser.add_argument("--strength", type=float, default=0.8, help="How much to transform the padded area.")
    
    args = parser.parse_args()

    model_id = "segmind/tiny-sd"
    
    print(f"Loading model {model_id}...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe = pipe.to("cpu")
    pipe.enable_attention_slicing()

    print("Loading and padding image...")
    init_image = load_image(args.image)
    
    # Pad the image
    # We use edge padding or a solid color to give the model something to work with
    padded_image = ImageOps.expand(init_image, border=args.padding, fill="gray")
    
    # Create mask for the padded area
    mask_image = Image.new("L", padded_image.size, 255) # 255 means change
    mask_image.paste(0, (args.padding, args.padding, args.padding + init_image.size[0], args.padding + init_image.size[1])) # 0 means keep

    # For Stable Diffusion, we should work with 512x512 blocks or resize carefully
    # To avoid distortion, we'll resize the padded image to 512x512 while keeping track of ratio
    # but for simplicity in a starter script, we'll just do a standard resize
    # and then resize back.
    original_size = padded_image.size
    input_image = padded_image.resize((512, 512))

    print(f"Processing outpainting via Img2Img...")
    generated_image = pipe(
        prompt=args.prompt, 
        image=input_image, 
        strength=args.strength, 
        num_inference_steps=args.steps
    ).images[0]

    # Resize generated back to padded size
    generated_image = generated_image.resize(original_size)

    # Blend: keep original center, use generated edges
    final_image = Image.composite(generated_image, padded_image, mask_image)

    final_image.save(args.output)
    print(f"Image saved to {args.output}")

if __name__ == "__main__":
    main()
