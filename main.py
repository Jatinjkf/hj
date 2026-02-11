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
from rich.markdown import Markdown
import requests
from io import BytesIO
import utils

# Suppress warnings
utils.suppress_warnings()

console = Console()

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

def get_pipeline(pipe_type="txt2img"):
    model_id = "segmind/tiny-sd"
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description=f"Loading Tiny-SD ({pipe_type})...", total=None)

        PipelineClass = StableDiffusionPipeline if pipe_type == "txt2img" else StableDiffusionImg2ImgPipeline

        try:
            pipe = PipelineClass.from_pretrained(model_id, torch_dtype=torch.float32, use_safetensors=True)
        except Exception as e:
            console.print(f"[yellow]Safetensors load failed ({e}), falling back to standard weights...[/yellow]")
            pipe = PipelineClass.from_pretrained(model_id, torch_dtype=torch.float32, use_safetensors=False)
        
        pipe = pipe.to("cpu")
        pipe.enable_attention_slicing()
        pipe.scheduler = utils.get_optimal_scheduler(pipe)

    return pipe

def run_txt2img():
    console.print(Panel("[bold cyan]Text-to-Image Mode[/bold cyan]"))
    prompt = Prompt.ask("Enter your prompt", default="A magical forest at night")
    steps = IntPrompt.ask("Inference steps", default=15)
    output = Prompt.ask("Output filename", default="output_txt2img.png")
    output = utils.ensure_extension(output)
    
    pipe = get_pipeline("txt2img")
    
    with Progress(SpinnerColumn(), TextColumn("Generating image..."), transient=True) as progress:
        progress.add_task("gen", total=None)
        image = pipe(prompt=prompt, num_inference_steps=steps).images[0]
    
    image.save(output)
    console.print(f"[bold green]Success![/bold green] Image saved to [bold white]{output}[/bold white]")

def run_img2img():
    console.print(Panel("[bold cyan]Image-to-Image Mode[/bold cyan]"))
    img_path = Prompt.ask("Enter path to input image")
    init_image = load_image(img_path)
    if not init_image: return

    prompt = Prompt.ask("Enter prompt for transformation", default="Cyberpunk style")
    strength = FloatPrompt.ask("Transformation strength (0.0 - 1.0)", default=0.75)
    steps = IntPrompt.ask("Inference steps", default=15)
    output = Prompt.ask("Output filename", default="output_img2img.png")
    output = utils.ensure_extension(output)
    
    pipe = get_pipeline("img2img")
    init_image = init_image.resize((512, 512))
    
    with Progress(SpinnerColumn(), TextColumn("Processing image..."), transient=True) as progress:
        progress.add_task("gen", total=None)
        image = pipe(prompt=prompt, image=init_image, strength=strength, num_inference_steps=steps).images[0]
    
    image.save(output)
    console.print(f"[bold green]Success![/bold green] Image saved to [bold white]{output}[/bold white]")

def run_inpaint():
    console.print(Panel("[bold cyan]Inpainting Mode[/bold cyan]"))
    img_path = Prompt.ask("Enter path to input image")
    mask_path = Prompt.ask("Enter path to mask image (white = area to change)")
    
    init_image = load_image(img_path)
    mask_image = load_image(mask_path)
    if not init_image or not mask_image: return

    prompt = Prompt.ask("Enter prompt for inpainting")
    strength = FloatPrompt.ask("Strength", default=0.75)
    output = Prompt.ask("Output filename", default="output_inpaint.png")
    output = utils.ensure_extension(output)
    
    pipe = get_pipeline("img2img")
    
    init_image = init_image.resize((512, 512))
    mask_image = mask_image.convert("L").resize((512, 512))
    
    with Progress(SpinnerColumn(), TextColumn("Inpainting..."), transient=True) as progress:
        progress.add_task("gen", total=None)
        generated = pipe(prompt=prompt, image=init_image, strength=strength).images[0]
        final = Image.composite(generated, init_image, mask_image)
    
    final.save(output)
    console.print(f"[bold green]Success![/bold green] Inpainted image saved to [bold white]{output}[/bold white]")

def run_outpaint():
    console.print(Panel("[bold cyan]Outpainting Mode[/bold cyan]"))
    img_path = Prompt.ask("Enter path to input image")
    init_image = load_image(img_path)
    if not init_image: return

    prompt = Prompt.ask("Enter prompt describing the full scene")
    padding = IntPrompt.ask("Padding pixels", default=128)
    output = Prompt.ask("Output filename", default="output_outpaint.png")
    output = utils.ensure_extension(output)
    
    pipe = get_pipeline("img2img")
    
    padded_image = ImageOps.expand(init_image, border=padding, fill="gray")
    mask_image = Image.new("L", padded_image.size, 255)
    mask_image.paste(0, (padding, padding, padding + init_image.size[0], padding + init_image.size[1]))
    
    orig_size = padded_image.size
    input_image = padded_image.resize((512, 512))
    
    with Progress(SpinnerColumn(), TextColumn("Outpainting..."), transient=True) as progress:
        progress.add_task("gen", total=None)
        generated = pipe(prompt=prompt, image=input_image, strength=0.8).images[0]
        generated = generated.resize(orig_size)
        final = Image.composite(generated, padded_image, mask_image)
    
    final.save(output)
    console.print(f"[bold green]Success![/bold green] Outpainted image saved to [bold white]{output}[/bold white]")

def show_menu():
    table = Table(title="[bold magenta]Tiny-SD Studio[/bold magenta]", show_header=True, header_style="bold cyan")
    table.add_column("Option", style="dim", width=6)
    table.add_column("Task")
    table.add_row("1", "Text to Image")
    table.add_row("2", "Image to Image")
    table.add_row("3", "Inpainting (Workaround)")
    table.add_row("4", "Outpainting")
    table.add_row("0", "Exit")
    console.print(table)

def main():
    console.print(Panel.fit(
        "[bold cyan]Welcome to the Tiny-SD CLI Studio[/bold cyan]\n[italic white]Distilled Power on your CPU[/italic white]",
        border_style="magenta"
    ))
    
    while True:
        show_menu()
        choice = Prompt.ask("Select an option", choices=["0", "1", "2", "3", "4"], default="1")
        
        if choice == "1": run_txt2img()
        elif choice == "2": run_img2img()
        elif choice == "3": run_inpaint()
        elif choice == "4": run_outpaint()
        elif choice == "0":
            console.print("[bold yellow]Goodbye![/bold yellow]")
            break
        console.print("\n" + "-"*30 + "\n")

if __name__ == "__main__":
    main()
