import torch
from diffusers import StableDiffusionPipeline
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate image with LoRA using Tiny-SD on CPU.")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt to generate.")
    parser.add_argument("--lora", type=str, required=True, help="Path or HF ID to the LoRA weights.")
    parser.add_argument("--output", type=str, default="output_lora.png", help="Output file name.")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps.")
    parser.add_argument("--scale", type=float, default=1.0, help="LoRA scale.")
    
    args = parser.parse_args()

    model_id = "segmind/tiny-sd"
    
    print(f"Loading base model {model_id}...")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    
    print(f"Loading LoRA weights from {args.lora}...")
    pipe.load_lora_weights(args.lora)
    
    pipe = pipe.to("cpu")
    pipe.enable_attention_slicing()

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
