import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
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
    parser = argparse.ArgumentParser(description="Generate variations of an image using Tiny-SD on CPU.")
    parser.add_argument("--image", type=str, required=True, help="Path or URL to the input image.")
    parser.add_argument("--prompt", type=str, default="", help="Optional prompt to guide variations.")
    parser.add_argument("--num_variations", type=int, default=3, help="Number of variations to generate.")
    parser.add_argument("--strength", type=float, default=0.5, help="How different the variations should be (0.1 to 0.9).")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps.")
    
    args = parser.parse_args()

    model_id = "segmind/tiny-sd"
    
    print(f"Loading model {model_id}...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe = pipe.to("cpu")
    pipe.enable_attention_slicing()

    print(f"Loading image from {args.image}...")
    init_image = load_image(args.image).resize((512, 512))

    for i in range(args.num_variations):
        print(f"Generating variation {i+1}/{args.num_variations}...")
        image = pipe(
            prompt=args.prompt, 
            image=init_image, 
            strength=args.strength, 
            num_inference_steps=args.steps
        ).images[0]
        
        output_name = f"variation_{i+1}.png"
        image.save(output_name)
        print(f"Variation saved to {output_name}")

if __name__ == "__main__":
    main()
