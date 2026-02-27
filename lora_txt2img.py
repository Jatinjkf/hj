import torch
from diffusers import StableDiffusionPipeline
import argparse
import utils

# Suppress warnings
utils.suppress_warnings()

def main():
    parser = argparse.ArgumentParser(description="Generate image with LoRA using Tiny-SD on CPU.")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt to generate.")
    parser.add_argument("--lora", type=str, required=True, help="Path or HF ID to the LoRA weights.")
    parser.add_argument("--output", type=str, default="output_lora.png", help="Output file name.")
    parser.add_argument("--steps", type=int, default=15, help="Number of inference steps (Default: 15).")
    parser.add_argument("--scale", type=float, default=1.0, help="LoRA scale.")
    
    args = parser.parse_args()
    args.output = utils.ensure_extension(args.output)

    model_id = "segmind/tiny-sd"
    
    print(f"Loading base model {model_id}...")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32, use_safetensors=True)
    except Exception as e:
         print(f"Safetensors load failed, falling back to standard weights...")
         pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32, use_safetensors=False)
    
    print(f"Loading LoRA weights from {args.lora}...")
    try:
        pipe.load_lora_weights(args.lora)
    except Exception as e:
        print(f"Error loading LoRA: {e}")
        return
    
    pipe = pipe.to("cpu")
    pipe.enable_attention_slicing()
    pipe.scheduler = utils.get_optimal_scheduler(pipe)

    print(f"Generating image with LoRA...")
    image = pipe(
        prompt=args.prompt, 
        num_inference_steps=args.steps,
        cross_attention_kwargs={"scale": args.scale} if args.scale != 1.0 else None
    ).images[0]

    image.save(args.output)
    print(f"Image saved to {args.output}")

if __name__ == "__main__":
    main()
