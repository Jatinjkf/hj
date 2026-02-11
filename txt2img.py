import torch
from diffusers import StableDiffusionPipeline
import argparse
import utils

# Suppress warnings
utils.suppress_warnings()

def main():
    parser = argparse.ArgumentParser(description="Generate image from text using Tiny-SD on CPU.")
    parser.add_argument("--prompt", type=str, default="A beautiful landscape with mountains and a lake", help="The prompt to generate.")
    parser.add_argument("--output", type=str, default="output_txt2img.png", help="Output file name.")
    parser.add_argument("--steps", type=int, default=15, help="Number of inference steps (Default: 15).")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale.")
    
    args = parser.parse_args()
    args.output = utils.ensure_extension(args.output)

    model_id = "segmind/tiny-sd"
    
    print(f"Loading model {model_id}...")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32, use_safetensors=True)
    except Exception as e:
        print(f"Safetensors load failed, falling back to standard weights...")
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32, use_safetensors=False)

    pipe = pipe.to("cpu")
    # Optimize for CPU/Low memory
    pipe.enable_attention_slicing()
    pipe.scheduler = utils.get_optimal_scheduler(pipe)

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
