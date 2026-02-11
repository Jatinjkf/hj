import torch
from diffusers import StableDiffusionPipeline
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate image from text using Tiny-SD on CPU.")
    parser.add_argument("--prompt", type=str, default="A beautiful landscape with mountains and a lake", help="The prompt to generate.")
    parser.add_argument("--output", type=str, default="output_txt2img.png", help="Output file name.")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale.")
    
    args = parser.parse_args()

    model_id = "segmind/tiny-sd"
    
    print(f"Loading model {model_id}...")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe = pipe.to("cpu")
    # Optimize for CPU/Low memory
    pipe.enable_attention_slicing()

    print(f"Generating image for prompt: '{args.prompt}'...")
    image = pipe(
        prompt=args.prompt, 
        num_inference_steps=args.steps, 
        guidance_scale=args.guidance_scale
    ).images[0]

    image.save(args.output)
    print(f"Image saved to {args.output}")

if __name__ == "__main__":
    main()
