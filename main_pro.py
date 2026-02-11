import os
import torch
import argparse
from PIL import Image, ImageOps
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, FloatPrompt
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import requests
from io import BytesIO
import utils
import shutil

# Suppress warnings
utils.suppress_warnings()

console = Console()

# Define paths
LOCAL_MODEL_DIR = "tiny-sd-models"
OV_MODEL_DIR = os.path.join(LOCAL_MODEL_DIR, "openvino")
PYTORCH_MODEL_DIR = os.path.join(LOCAL_MODEL_DIR, "pytorch")

os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
os.makedirs(OV_MODEL_DIR, exist_ok=True)
os.makedirs(PYTORCH_MODEL_DIR, exist_ok=True)

def detect_intel_gpu():
    """Detect Intel GPU availability via DirectML or XPU."""
    try:
        import torch_directml
        if torch_directml.is_available():
            return "directml"
    except ImportError:
        pass
    
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

def get_pipeline(pipe_type, device_key, width=512, height=512, batch_size=1, scheduler_name="DPM++ 2M Karras"):
    model_id = "segmind/tiny-sd"
    
    # Handle OpenVINO separately (CPU or GPU)
    if device_key in ["openvino_cpu", "openvino_gpu"]:
        try:
            from optimum.intel.openvino import OVStableDiffusionPipeline, OVStableDiffusionImg2ImgPipeline
        except ImportError:
            console.print("[bold red]Error:[/bold red] optimum-intel[openvino] not installed. Please install it to use OpenVINO.")
            return None

        ov_device = "GPU" if device_key == "openvino_gpu" else "CPU"

        # Check if local OpenVINO model exists
        # We need to check for model files, e.g., openvino_model.xml for unet/vae/text_encoder
        # A simple check is if the directory is populated.
        is_local_ov_available = os.path.exists(os.path.join(OV_MODEL_DIR, "unet", "openvino_model.xml"))

        load_message = f"Loading Tiny-SD (OpenVINO {ov_device})..."
        if is_local_ov_available:
            load_message += " [Local Cache]"

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description=load_message, total=None)

            try:
                if is_local_ov_available:
                    # Load from local directory
                    if pipe_type == "txt2img":
                        pipe = OVStableDiffusionPipeline.from_pretrained(OV_MODEL_DIR)
                    else:
                        pipe = OVStableDiffusionImg2ImgPipeline.from_pretrained(OV_MODEL_DIR)
                else:
                    # Conversion required
                    console.print("[yellow]First run: Downloading and converting model to OpenVINO... This may take a while.[/yellow]")
                    try:
                        # Try direct conversion first
                        if pipe_type == "txt2img":
                            pipe = OVStableDiffusionPipeline.from_pretrained(model_id, export=True, cache_dir=PYTORCH_MODEL_DIR)
                        else:
                            pipe = OVStableDiffusionImg2ImgPipeline.from_pretrained(model_id, export=True, cache_dir=PYTORCH_MODEL_DIR)
                    except Exception as e:
                        console.print(f"[yellow]Direct conversion failed ({e}), trying robust fallback via PyTorch...[/yellow]")
                        # Robust fallback: Load PT -> Convert -> Save
                        pt_pipe_cls = StableDiffusionPipeline if pipe_type == "txt2img" else StableDiffusionImg2ImgPipeline
                        # Load PyTorch model to local cache
                        pt_pipe = pt_pipe_cls.from_pretrained(model_id, torch_dtype=torch.float32, use_safetensors=False, cache_dir=PYTORCH_MODEL_DIR)

                        if pipe_type == "txt2img":
                            pipe = OVStableDiffusionPipeline.from_pipe(pt_pipe, export=True)
                        else:
                            # Use txt2img pipe for conversion usually safer, then cast?
                            # Or just use from_pipe on img2img class if supported.
                            pipe = OVStableDiffusionImg2ImgPipeline.from_pipe(pt_pipe, export=True)

                    # Save the converted model locally for next time
                    pipe.save_pretrained(OV_MODEL_DIR)
                    console.print(f"[bold green]Model converted and saved to {OV_MODEL_DIR}[/bold green]")

                # Configure scheduler BEFORE compilation
                utils.configure_scheduler(pipe, scheduler_name)

                # Reshape to static shape
                pipe.reshape(batch_size=1, height=height, width=width, num_images_per_prompt=batch_size)

                # Compile and move to device
                pipe.to(ov_device)
                pipe.compile()

            except Exception as e:
                console.print(f"[bold red]OpenVINO Load/Compile Error:[/bold red] {e}")
                console.print("[yellow]Try upgrading torch: `pip install --upgrade torch`[/yellow]")
                return None

        return pipe

    # Standard PyTorch / DirectML / XPU logic
    if device_key == "intel":
        intel_backend = detect_intel_gpu()
        if intel_backend == "directml":
            import torch_directml
            device = torch_directml.device()
            dtype = torch.float32
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
        
        PipelineClass = StableDiffusionPipeline if pipe_type == "txt2img" else StableDiffusionImg2ImgPipeline
        
        try:
            pipe = PipelineClass.from_pretrained(model_id, torch_dtype=dtype, use_safetensors=True, cache_dir=PYTORCH_MODEL_DIR)
        except Exception as e:
            console.print(f"[yellow]Safetensors load failed ({e}), falling back to standard weights...[/yellow]")
            pipe = PipelineClass.from_pretrained(model_id, torch_dtype=dtype, use_safetensors=False, cache_dir=PYTORCH_MODEL_DIR)

        pipe = pipe.to(device)
        pipe.enable_attention_slicing()
        # Configure scheduler
        utils.configure_scheduler(pipe, scheduler_name)
            
    return pipe

def get_user_inputs(task_name):
    """Collect common inputs for generation tasks."""
    prompt = Prompt.ask("Enter your prompt", default="A beautiful digital art of a sunset")
    neg_prompt = Prompt.ask("Enter negative prompt (optional)", default="")
    if not neg_prompt: neg_prompt = None

    # Scheduler Selection
    schedulers = utils.get_scheduler_list()
    console.print("[bold]Select Sampler:[/bold]")
    for idx, s in enumerate(schedulers):
        console.print(f"{idx+1}. {s}")

    sched_choice = IntPrompt.ask("Choose sampler", default=1, choices=[str(i+1) for i in range(len(schedulers))])
    scheduler_name = schedulers[sched_choice-1]

    steps = IntPrompt.ask("Inference steps", default=15)
    cfg_scale = FloatPrompt.ask("CFG Scale (Guidance)", default=7.0)
    seed = IntPrompt.ask("Seed (-1 for random)", default=-1)

    width = IntPrompt.ask("Width", default=512)
    height = IntPrompt.ask("Height", default=512)

    batch_size = IntPrompt.ask("Batch Size (Images per generation)", default=1)
    batch_count = IntPrompt.ask("Batch Count (Number of generations)", default=1)

    return {
        "prompt": prompt,
        "negative_prompt": neg_prompt,
        "scheduler": scheduler_name,
        "steps": steps,
        "guidance_scale": cfg_scale,
        "seed": seed,
        "width": width,
        "height": height,
        "batch_size": batch_size,
        "batch_count": batch_count
    }

def run_task(task_name, device_key):
    # Map selection to internal device key
    if device_key == "1": selected_device = "cpu"
    elif device_key == "2": selected_device = "intel"
    elif device_key == "3": selected_device = "openvino_cpu"
    elif device_key == "4": selected_device = "openvino_gpu"
    else: selected_device = device_key

    console.print(Panel(f"[bold cyan]{task_name}[/bold cyan] Mode (Running on [bold green]{selected_device.upper()}[/bold green])"))
    
    if task_name == "Text to Image":
        inputs = get_user_inputs(task_name)
        output_base = Prompt.ask("Output filename (base)", default="output_txt2img")
        
        pipe = get_pipeline(
            "txt2img",
            selected_device,
            width=inputs["width"],
            height=inputs["height"],
            batch_size=inputs["batch_size"],
            scheduler_name=inputs["scheduler"]
        )
        if not pipe: return

        # Loop for Batch Count
        total_images = 0
        for i in range(inputs["batch_count"]):
            # Set Seed
            current_seed = inputs["seed"] if inputs["seed"] != -1 else torch.randint(0, 2**32, (1,)).item()
            generator = torch.manual_seed(current_seed)

            console.print(f"[dim]Batch {i+1}/{inputs['batch_count']} | Seed: {current_seed}[/dim]")

            with Progress(SpinnerColumn(), TextColumn("Generating..."), transient=True) as progress:
                progress.add_task("gen", total=None)
                images = pipe(
                    prompt=inputs["prompt"],
                    negative_prompt=inputs["negative_prompt"],
                    num_inference_steps=inputs["steps"],
                    guidance_scale=inputs["guidance_scale"],
                    width=inputs["width"],
                    height=inputs["height"],
                    num_images_per_prompt=inputs["batch_size"],
                    generator=generator
                ).images

            # Save images
            for j, img in enumerate(images):
                suffix = f"_{total_images}.png" if (inputs["batch_count"] > 1 or inputs["batch_size"] > 1) else ".png"
                out_name = f"{output_base}{suffix}"
                img.save(out_name)
                total_images += 1

        console.print(f"[bold green]Success![/bold green] Saved {total_images} images.")
        
    elif task_name == "Image to Image":
        img_path = Prompt.ask("Enter path to input image")
        init_image = load_image(img_path)
        if not init_image: return

        # Reuse get_user_inputs but maybe add strength?
        inputs = get_user_inputs(task_name)
        strength = FloatPrompt.ask("Strength (0.1 - 0.9)", default=0.7)
        output_base = Prompt.ask("Output filename (base)", default="output_img2img")
        
        pipe = get_pipeline("img2img", selected_device, width=inputs["width"], height=inputs["height"], batch_size=inputs["batch_size"], scheduler_name=inputs["scheduler"])
        if not pipe: return

        init_image = init_image.resize((inputs["width"], inputs["height"])) # Resize input to match generation size

        total_images = 0
        for i in range(inputs["batch_count"]):
            current_seed = inputs["seed"] if inputs["seed"] != -1 else torch.randint(0, 2**32, (1,)).item()
            generator = torch.manual_seed(current_seed)

            with Progress(SpinnerColumn(), TextColumn("Processing..."), transient=True) as progress:
                progress.add_task("gen", total=None)
                images = pipe(
                    prompt=inputs["prompt"],
                    negative_prompt=inputs["negative_prompt"],
                    image=init_image,
                    strength=strength,
                    num_inference_steps=inputs["steps"],
                    guidance_scale=inputs["guidance_scale"],
                    num_images_per_prompt=inputs["batch_size"],
                    generator=generator
                ).images

            for j, img in enumerate(images):
                suffix = f"_{total_images}.png" if (inputs["batch_count"] > 1 or inputs["batch_size"] > 1) else ".png"
                out_name = f"{output_base}{suffix}"
                img.save(out_name)
                total_images += 1

        console.print(f"[bold green]Success![/bold green] Saved {total_images} images.")

    elif task_name == "Inpainting":
        # Simplified for now, just adding scheduler/seed support basically
        img_path = Prompt.ask("Enter path to input image")
        mask_path = Prompt.ask("Enter path to mask image")
        init_image = load_image(img_path)
        mask_image = load_image(mask_path)
        if not init_image or not mask_image: return
        
        inputs = get_user_inputs(task_name)
        output_base = Prompt.ask("Output filename (base)", default="output_inpaint")

        pipe = get_pipeline("img2img", selected_device, width=inputs["width"], height=inputs["height"], batch_size=inputs["batch_size"], scheduler_name=inputs["scheduler"])
        if not pipe: return

        init_image = init_image.resize((inputs["width"], inputs["height"]))
        mask_image = mask_image.convert("L").resize((inputs["width"], inputs["height"]))

        total_images = 0
        for i in range(inputs["batch_count"]):
            current_seed = inputs["seed"] if inputs["seed"] != -1 else torch.randint(0, 2**32, (1,)).item()
            generator = torch.manual_seed(current_seed)

            with Progress(SpinnerColumn(), TextColumn("Inpainting..."), transient=True) as progress:
                progress.add_task("gen", total=None)
                # Use img2img + composite workaround
                gens = pipe(
                    prompt=inputs["prompt"],
                    negative_prompt=inputs["negative_prompt"],
                    image=init_image,
                    strength=0.75,
                    num_inference_steps=inputs["steps"],
                    guidance_scale=inputs["guidance_scale"],
                    num_images_per_prompt=inputs["batch_size"],
                    generator=generator
                ).images

                for gen in gens:
                    final = Image.composite(gen, init_image, mask_image)
                    suffix = f"_{total_images}.png" if (inputs["batch_count"] > 1 or inputs["batch_size"] > 1) else ".png"
                    out_name = f"{output_base}{suffix}"
                    final.save(out_name)
                    total_images += 1

        console.print(f"[bold green]Success![/bold green] Saved {total_images} images.")

    elif task_name == "Outpainting":
        # Similar updates...
        img_path = Prompt.ask("Enter path to input image")
        init_image = load_image(img_path)
        if not init_image: return

        inputs = get_user_inputs(task_name)
        padding = IntPrompt.ask("Padding pixels", default=128)
        output_base = Prompt.ask("Output filename (base)", default="output_outpaint")
        
        pipe = get_pipeline("img2img", selected_device, width=512, height=512, batch_size=inputs["batch_size"], scheduler_name=inputs["scheduler"]) # Force 512 for outpaint logic simplicity
        if not pipe: return

        padded_image = ImageOps.expand(init_image, border=padding, fill="gray")
        mask_image = Image.new("L", padded_image.size, 255)
        mask_image.paste(0, (padding, padding, padding + init_image.size[0], padding + init_image.size[1]))
        
        orig_size = padded_image.size
        input_image = padded_image.resize((512, 512)) # Hardcoded logic in previous steps, keeping it simple

        total_images = 0
        for i in range(inputs["batch_count"]):
            current_seed = inputs["seed"] if inputs["seed"] != -1 else torch.randint(0, 2**32, (1,)).item()
            generator = torch.manual_seed(current_seed)

            with Progress(SpinnerColumn(), TextColumn("Outpainting..."), transient=True) as progress:
                progress.add_task("gen", total=None)
                gens = pipe(
                    prompt=inputs["prompt"],
                    negative_prompt=inputs["negative_prompt"],
                    image=input_image,
                    strength=0.8,
                    num_inference_steps=inputs["steps"],
                    guidance_scale=inputs["guidance_scale"],
                    num_images_per_prompt=inputs["batch_size"],
                    generator=generator
                ).images

                for gen in gens:
                    gen = gen.resize(orig_size)
                    final = Image.composite(gen, padded_image, mask_image)
                    suffix = f"_{total_images}.png" if (inputs["batch_count"] > 1 or inputs["batch_size"] > 1) else ".png"
                    out_name = f"{output_base}{suffix}"
                    final.save(out_name)
                    total_images += 1
        console.print(f"[bold green]Success![/bold green] Saved {total_images} images.")

    elif task_name == "LoRA Text-to-Image":
        inputs = get_user_inputs(task_name)
        lora_path = Prompt.ask("Enter path or HF ID to LoRA")
        scale = FloatPrompt.ask("LoRA scale", default=1.0)
        output_base = Prompt.ask("Output filename (base)", default="output_lora")

        pipe = get_pipeline("txt2img", selected_device, width=inputs["width"], height=inputs["height"], batch_size=inputs["batch_size"], scheduler_name=inputs["scheduler"])
        if not pipe: return

        with Progress(SpinnerColumn(), TextColumn("Loading LoRA..."), transient=True) as progress:
             progress.add_task("gen", total=None)
             try:
                 if hasattr(pipe, "load_lora_weights"):
                     pipe.load_lora_weights(lora_path)
                 else:
                     console.print("[bold red]Error:[/bold red] This version of Optimum Intel might not support LoRA loading directly.")
                     return
             except Exception as e:
                 console.print(f"[bold red]Error loading LoRA:[/bold red] {e}")
                 return

        total_images = 0
        for i in range(inputs["batch_count"]):
            current_seed = inputs["seed"] if inputs["seed"] != -1 else torch.randint(0, 2**32, (1,)).item()
            generator = torch.manual_seed(current_seed)

            with Progress(SpinnerColumn(), TextColumn("Generating..."), transient=True) as progress:
                progress.add_task("gen", total=None)
                if "openvino" in selected_device:
                    images = pipe(
                        prompt=inputs["prompt"],
                        negative_prompt=inputs["negative_prompt"],
                        num_inference_steps=inputs["steps"],
                        guidance_scale=inputs["guidance_scale"],
                        width=inputs["width"],
                        height=inputs["height"],
                        num_images_per_prompt=inputs["batch_size"],
                        generator=generator
                    ).images
                else:
                    images = pipe(
                        prompt=inputs["prompt"],
                        negative_prompt=inputs["negative_prompt"],
                        num_inference_steps=inputs["steps"],
                        guidance_scale=inputs["guidance_scale"],
                        width=inputs["width"],
                        height=inputs["height"],
                        num_images_per_prompt=inputs["batch_size"],
                        generator=generator,
                        cross_attention_kwargs={"scale": scale} if scale != 1.0 else None
                    ).images

            for img in images:
                suffix = f"_{total_images}.png" if (inputs["batch_count"] > 1 or inputs["batch_size"] > 1) else ".png"
                out_name = f"{output_base}{suffix}"
                img.save(out_name)
                total_images += 1

        console.print(f"[bold green]Success![/bold green] Saved {total_images} images.")

    elif task_name == "Variations":
        img_path = Prompt.ask("Enter path to input image")
        init_image = load_image(img_path)
        if not init_image: return

        inputs = get_user_inputs(task_name)
        strength = FloatPrompt.ask("Strength (0.1 - 0.9)", default=0.5)

        # Override batch counts? Or just use them.
        # Usually variations imply we want batch_count * batch_size variations.

        pipe = get_pipeline("img2img", selected_device, width=inputs["width"], height=inputs["height"], batch_size=inputs["batch_size"], scheduler_name=inputs["scheduler"])
        if not pipe: return

        init_image = init_image.resize((inputs["width"], inputs["height"]))

        total_images = 0
        for i in range(inputs["batch_count"]):
            current_seed = inputs["seed"] if inputs["seed"] != -1 else torch.randint(0, 2**32, (1,)).item()
            generator = torch.manual_seed(current_seed)

            with Progress(SpinnerColumn(), TextColumn("Generating Variations..."), transient=True) as progress:
                progress.add_task("gen", total=None)
                # Empty prompt allows the image to guide the generation entirely (with strength)
                images = pipe(
                    prompt="",
                    negative_prompt=inputs["negative_prompt"],
                    image=init_image,
                    strength=strength,
                    num_inference_steps=inputs["steps"],
                    guidance_scale=inputs["guidance_scale"],
                    num_images_per_prompt=inputs["batch_size"],
                    generator=generator
                ).images

                for img in images:
                    suffix = f"_{total_images}.png"
                    out_name = f"variation_{total_images+1}.png"
                    img.save(out_name)
                    total_images += 1
                    console.print(f"Saved {out_name}")

        console.print(f"[bold green]All variations saved![/bold green]")


def main():
    console.print(Panel.fit(
        "[bold cyan]Tiny-SD Studio Pro[/bold cyan]\n[italic white]Optimized for Intel OpenVINO, CPU & HD 620[/italic white]",
        border_style="magenta"
    ))
    
    # Device Selection
    device_table = Table(title="Select Computing Hardware", show_header=True)
    device_table.add_column("Key", style="cyan")
    device_table.add_column("Hardware")
    device_table.add_row("1", "CPU (Standard PyTorch)")
    device_table.add_row("2", "Intel GPU (DirectML/XPU)")
    device_table.add_row("3", "Intel OpenVINO CPU (Fast)")
    device_table.add_row("4", "Intel OpenVINO GPU (Fastest for iGPU)")
    console.print(device_table)
    
    choice = Prompt.ask("Choose your hardware", choices=["1", "2", "3", "4"], default="3")

    # Map to internal key
    if choice == "1": selected_device = "cpu"
    elif choice == "2": selected_device = "intel"
    elif choice == "3": selected_device = "openvino_cpu"
    else: selected_device = "openvino_gpu"
    
    while True:
        table = Table(title="[bold magenta]Generation Menu[/bold magenta]")
        table.add_column("Option", width=6)
        table.add_column("Task")
        table.add_row("1", "Text to Image")
        table.add_row("2", "Image to Image")
        table.add_row("3", "Inpainting")
        table.add_row("4", "Outpainting")
        table.add_row("5", "LoRA Text-to-Image")
        table.add_row("6", "Variations")
        table.add_row("C", "Change Hardware")
        table.add_row("0", "Exit")
        console.print(table)
        
        choice = Prompt.ask("Select an option", choices=["0", "1", "2", "3", "4", "5", "6", "C"], default="1")
        
        if choice == "0":
            console.print("[bold yellow]Goodbye![/bold yellow]")
            break
        elif choice == "C":
            new_choice = Prompt.ask("Choose hardware", choices=["1", "2", "3", "4"])
            if new_choice == "1": selected_device = "cpu"
            elif new_choice == "2": selected_device = "intel"
            elif new_choice == "3": selected_device = "openvino_cpu"
            else: selected_device = "openvino_gpu"
        else:
            tasks = {
                "1": "Text to Image",
                "2": "Image to Image",
                "3": "Inpainting",
                "4": "Outpainting",
                "5": "LoRA Text-to-Image",
                "6": "Variations"
            }
            run_task(tasks[choice], selected_device)
        
        console.print("\n" + "="*40 + "\n")

if __name__ == "__main__":
    main()
