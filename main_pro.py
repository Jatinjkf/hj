import os
import torch
import argparse
from PIL import Image, ImageOps
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, FloatPrompt
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import requests
from io import BytesIO

console = Console()

def detect_intel_gpu():
    """Detect Intel GPU availability via DirectML or XPU."""
    # Check for DirectML (Standard for Intel HD 620 on Windows)
    try:
        import torch_directml
        if torch_directml.is_available():
            return "directml"
    except ImportError:
        pass
    
    # Check for XPU (Newer Intel official support)
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
    except:
        pass
        
    return None

def load_image(image_path):
    try:
        if image_path.startswith("http"):
            response = requests.get(image_path)
            return Image.open(BytesIO(response.content)).convert("RGB")
        else:
            return Image.open(image_path).convert("RGB")
    except Exception as e:
        console.print(f"[bold red]Error loading image:[/bold red] {e}")
        return None

def get_pipeline(pipe_type, device_key):
    model_id = "segmind/tiny-sd"
    
    if device_key == "intel":
        # Determine which backend we are actually using
        intel_backend = detect_intel_gpu()
        if intel_backend == "directml":
            import torch_directml
            device = torch_directml.device()
            dtype = torch.float32 # DirectML requires float32 for many ops
        elif intel_backend == "xpu":
            device = torch.device("xpu")
            dtype = torch.float16
        else:
            console.print("[yellow]Warning: Intel GPU not detected. Falling back to CPU.[/yellow]")
            device = torch.device("cpu")
            dtype = torch.float32
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description=f"Loading Tiny-SD on {device_key.upper()}...", total=None)
        
        if pipe_type == "txt2img":
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
        else:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=dtype)
        
        pipe = pipe.to(device)
        pipe.enable_attention_slicing()
            
    return pipe

def run_task(task_name, device_key):
    console.print(Panel(f"[bold cyan]{task_name}[/bold cyan] Mode (Running on [bold green]{device_key.upper()}[/bold green])"))
    
    if task_name == "Text to Image":
        prompt = Prompt.ask("Enter your prompt", default="A beautiful digital art of a sunset")
        steps = IntPrompt.ask("Inference steps", default=20)
        output = Prompt.ask("Output filename", default="output_txt2img.png")
        
        pipe = get_pipeline("txt2img", device_key)
        with Progress(SpinnerColumn(), TextColumn("Generating..."), transient=True) as progress:
            progress.add_task("gen", total=None)
            image = pipe(prompt=prompt, num_inference_steps=steps).images[0]
        image.save(output)
        
    elif task_name == "Image to Image":
        img_path = Prompt.ask("Enter path to input image")
        init_image = load_image(img_path)
        if not init_image: return
        prompt = Prompt.ask("Enter prompt for transformation", default="Cyberpunk style")
        strength = FloatPrompt.ask("Strength (0.1 - 0.9)", default=0.7)
        output = Prompt.ask("Output filename", default="output_img2img.png")
        
        pipe = get_pipeline("img2img", device_key)
        init_image = init_image.resize((512, 512))
        with Progress(SpinnerColumn(), TextColumn("Processing..."), transient=True) as progress:
            progress.add_task("gen", total=None)
            image = pipe(prompt=prompt, image=init_image, strength=strength).images[0]
        image.save(output)

    elif task_name == "Inpainting":
        img_path = Prompt.ask("Enter path to input image")
        mask_path = Prompt.ask("Enter path to mask image")
        init_image = load_image(img_path)
        mask_image = load_image(mask_path)
        if not init_image or not mask_image: return
        prompt = Prompt.ask("Enter prompt for inpainting")
        output = Prompt.ask("Output filename", default="output_inpaint.png")
        
        pipe = get_pipeline("img2img", device_key)
        init_image = init_image.resize((512, 512))
        mask_image = mask_image.convert("L").resize((512, 512))
        with Progress(SpinnerColumn(), TextColumn("Inpainting..."), transient=True) as progress:
            progress.add_task("gen", total=None)
            gen = pipe(prompt=prompt, image=init_image, strength=0.75).images[0]
            final = Image.composite(gen, init_image, mask_image)
        final.save(output)

    elif task_name == "Outpainting":
        img_path = Prompt.ask("Enter path to input image")
        init_image = load_image(img_path)
        if not init_image: return
        prompt = Prompt.ask("Enter prompt describing the full scene")
        padding = IntPrompt.ask("Padding pixels", default=128)
        output = Prompt.ask("Output filename", default="output_outpaint.png")
        
        pipe = get_pipeline("img2img", device_key)
        padded_image = ImageOps.expand(init_image, border=padding, fill="gray")
        mask_image = Image.new("L", padded_image.size, 255)
        mask_image.paste(0, (padding, padding, padding + init_image.size[0], padding + init_image.size[1]))
        
        orig_size = padded_image.size
        input_image = padded_image.resize((512, 512))
        with Progress(SpinnerColumn(), TextColumn("Outpainting..."), transient=True) as progress:
            progress.add_task("gen", total=None)
            gen = pipe(prompt=prompt, image=input_image, strength=0.8).images[0]
            gen = gen.resize(orig_size)
            final = Image.composite(gen, padded_image, mask_image)
        final.save(output)

    console.print(f"[bold green]Success![/bold green] File saved.")

def main():
    console.print(Panel.fit(
        "[bold cyan]Tiny-SD Studio Pro[/bold cyan]\n[italic white]Optimized for CPU & Intel HD 620[/italic white]",
        border_style="magenta"
    ))
    
    # Simple Device Selection
    device_table = Table(title="Select Computing Hardware", show_header=True)
    device_table.add_column("Key", style="cyan")
    device_table.add_column("Hardware")
    device_table.add_row("1", "CPU (Standard)")
    device_table.add_row("2", "Intel Integrated GPU (HD 620)")
    console.print(device_table)
    
    choice = Prompt.ask("Choose your hardware", choices=["1", "2"], default="1")
    selected_device = "cpu" if choice == "1" else "intel"
    
    while True:
        table = Table(title="[bold magenta]Generation Menu[/bold magenta]")
        table.add_column("Option", width=6)
        table.add_column("Task")
        table.add_row("1", "Text to Image")
        table.add_row("2", "Image to Image")
        table.add_row("3", "Inpainting")
        table.add_row("4", "Outpainting")
        table.add_row("C", "Change Hardware")
        table.add_row("0", "Exit")
        console.print(table)
        
        choice = Prompt.ask("Select an option", choices=["0", "1", "2", "3", "4", "C"], default="1")
        
        if choice == "0":
            console.print("[bold yellow]Goodbye![/bold yellow]")
            break
        elif choice == "C":
            new_choice = Prompt.ask("Choose hardware", choices=["1", "2"])
            selected_device = "cpu" if new_choice == "1" else "intel"
        else:
            tasks = {"1": "Text to Image", "2": "Image to Image", "3": "Inpainting", "4": "Outpainting"}
            run_task(tasks[choice], selected_device)
        
        console.print("\n" + "="*40 + "\n")

if __name__ == "__main__":
    main()
