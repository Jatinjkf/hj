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
import model_manager
import lora_manager
import hashlib

# Suppress warnings
utils.suppress_warnings()

console = Console()

# Define paths
LOCAL_MODEL_DIR = "Models"
OV_CACHE_DIR = os.path.join(LOCAL_MODEL_DIR, "openvino_cache")

os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
os.makedirs(OV_CACHE_DIR, exist_ok=True)

# Global cache for pipeline
CACHED_PIPELINE = None
CACHED_CONFIG = {}

# Global State
CURRENT_MODEL_PATH = "segmind/tiny-sd" # Default HF ID
CURRENT_MODEL_TYPE = "hf_id"
CURRENT_LORA_PATH = None
CURRENT_LORA_SCALE = 1.0

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
    global CACHED_PIPELINE, CACHED_CONFIG, CURRENT_MODEL_PATH, CURRENT_MODEL_TYPE, CURRENT_LORA_PATH, CURRENT_LORA_SCALE

    # Check if we can reuse the cached pipeline
    if CACHED_PIPELINE is not None and \
       CACHED_CONFIG.get("device_key") == device_key and \
       CACHED_CONFIG.get("model_path") == CURRENT_MODEL_PATH and \
       CACHED_CONFIG.get("lora_path") == CURRENT_LORA_PATH and \
       CACHED_CONFIG.get("lora_scale") == CURRENT_LORA_SCALE:

        if device_key in ["openvino_cpu", "openvino_gpu"]:
            current_dims = (CACHED_CONFIG.get("width"), CACHED_CONFIG.get("height"), CACHED_CONFIG.get("batch_size"))
            new_dims = (width, height, batch_size)

            if current_dims == new_dims:
                utils.configure_scheduler(CACHED_PIPELINE, scheduler_name)
                return CACHED_PIPELINE
            else:
                console.print(f"[dim]Reshaping OpenVINO model from {current_dims} to {new_dims}...[/dim]")
                try:
                    CACHED_PIPELINE.reshape(batch_size=1, height=height, width=width, num_images_per_prompt=batch_size)
                    ov_device = "GPU" if device_key == "openvino_gpu" else "CPU"
                    CACHED_PIPELINE.compile()
                    CACHED_CONFIG.update({"width": width, "height": height, "batch_size": batch_size})
                    utils.configure_scheduler(CACHED_PIPELINE, scheduler_name)
                    return CACHED_PIPELINE
                except Exception as e:
                    console.print(f"[yellow]Reshape/Compile failed ({e}), reloading pipeline...[/yellow]")
        else:
            utils.configure_scheduler(CACHED_PIPELINE, scheduler_name)
            return CACHED_PIPELINE

    # ----------------------------------------
    # Load New Pipeline
    # ----------------------------------------

    model_name_clean = os.path.basename(CURRENT_MODEL_PATH).replace(".safetensors", "").replace(".ckpt", "").replace(":", "_")
    if CURRENT_MODEL_TYPE == "hf_id":
        model_name_clean = CURRENT_MODEL_PATH.split("/")[-1]

    # OpenVINO Handling
    if device_key in ["openvino_cpu", "openvino_gpu"]:
        model_hash = hashlib.md5(CURRENT_MODEL_PATH.encode()).hexdigest()[:8]
        model_ov_dir = os.path.join(LOCAL_MODEL_DIR, f"{model_name_clean}_{model_hash}_openvino")

        try:
            from optimum.intel.openvino import OVStableDiffusionPipeline, OVStableDiffusionImg2ImgPipeline
        except ImportError:
            return None

        ov_device = "GPU" if device_key == "openvino_gpu" else "CPU"
        is_ov_cached = os.path.exists(os.path.join(model_ov_dir, "model_index.json"))

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
            progress.add_task(description=f"Loading {model_name_clean} (OpenVINO {ov_device})...", total=None)
            try:
                ov_config = {"CACHE_DIR": OV_CACHE_DIR}

                # Load Base Model
                if is_ov_cached:
                    if pipe_type == "txt2img":
                        pipe = OVStableDiffusionPipeline.from_pretrained(model_ov_dir, ov_config=ov_config)
                    else:
                        pipe = OVStableDiffusionImg2ImgPipeline.from_pretrained(model_ov_dir, ov_config=ov_config)
                else:
                    console.print(f"[yellow]Converting {model_name_clean} to OpenVINO... (First time only)[/yellow]")
                    if CURRENT_MODEL_TYPE == "file":
                        pipe = OVStableDiffusionPipeline.from_single_file(CURRENT_MODEL_PATH, export=True, ov_config=ov_config)
                    elif CURRENT_MODEL_TYPE == "folder":
                        pipe = OVStableDiffusionPipeline.from_pretrained(CURRENT_MODEL_PATH, export=True, ov_config=ov_config)
                    else:
                        try:
                            pipe = OVStableDiffusionPipeline.from_pretrained(CURRENT_MODEL_PATH, export=True, ov_config=ov_config)
                        except:
                             console.print(f"[yellow]Direct conversion failed, trying robust fallback...[/yellow]")
                             pt_pipe = StableDiffusionPipeline.from_pretrained(CURRENT_MODEL_PATH, use_safetensors=False)
                             pipe = OVStableDiffusionPipeline.from_pipe(pt_pipe, export=True, ov_config=ov_config)
                    pipe.save_pretrained(model_ov_dir)

                # Apply LoRA if selected
                if CURRENT_LORA_PATH:
                    console.print(f"[cyan]Applying LoRA: {os.path.basename(CURRENT_LORA_PATH)} (Scale: {CURRENT_LORA_SCALE})[/cyan]")
                    try:
                        # OpenVINO usually fuses LoRA.
                        if hasattr(pipe, "load_lora_weights"):
                            pipe.load_lora_weights(CURRENT_LORA_PATH)
                    except Exception as e:
                        console.print(f"[red]Failed to apply LoRA: {e}[/red]")

                utils.configure_scheduler(pipe, scheduler_name)
                pipe.reshape(batch_size=1, height=height, width=width, num_images_per_prompt=batch_size)
                pipe.to(ov_device)
                pipe.compile()

                CACHED_PIPELINE = pipe
                CACHED_CONFIG = {
                    "device_key": device_key,
                    "model_path": CURRENT_MODEL_PATH,
                    "lora_path": CURRENT_LORA_PATH,
                    "lora_scale": CURRENT_LORA_SCALE,
                    "width": width, "height": height, "batch_size": batch_size
                }

            except Exception as e:
                console.print(f"[bold red]OpenVINO Error:[/bold red] {e}")
                return None
        return pipe

    # PyTorch Handling
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
            device = torch.device("cpu")
            dtype = torch.float32
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task(description=f"Loading {model_name_clean} (PyTorch)...", total=None)
        
        PipelineClass = StableDiffusionPipeline if pipe_type == "txt2img" else StableDiffusionImg2ImgPipeline
        
        try:
            if CURRENT_MODEL_TYPE == "file":
                pipe = PipelineClass.from_single_file(CURRENT_MODEL_PATH, torch_dtype=dtype)
            elif CURRENT_MODEL_TYPE == "folder":
                pipe = PipelineClass.from_pretrained(CURRENT_MODEL_PATH, torch_dtype=dtype, use_safetensors=True)
            else:
                try:
                    pipe = PipelineClass.from_pretrained(CURRENT_MODEL_PATH, torch_dtype=dtype, use_safetensors=True)
                except:
                    pipe = PipelineClass.from_pretrained(CURRENT_MODEL_PATH, torch_dtype=dtype, use_safetensors=False)
        except Exception as e:
            console.print(f"[bold red]Load failed:[/bold red] {e}")
            return None

        # Apply LoRA for PyTorch
        if CURRENT_LORA_PATH:
             console.print(f"[cyan]Applying LoRA: {os.path.basename(CURRENT_LORA_PATH)}[/cyan]")
             try:
                 pipe.load_lora_weights(CURRENT_LORA_PATH)
             except Exception as e:
                 console.print(f"[red]LoRA Error:[/red] {e}")

        pipe = pipe.to(device)
        pipe.enable_attention_slicing()
        utils.configure_scheduler(pipe, scheduler_name)

        CACHED_PIPELINE = pipe
        CACHED_CONFIG = {
            "device_key": device_key,
            "model_path": CURRENT_MODEL_PATH,
            "lora_path": CURRENT_LORA_PATH,
            "lora_scale": CURRENT_LORA_SCALE,
            "width": width, "height": height, "batch_size": batch_size
        }
            
    return pipe

def get_common_settings():
    """Collect common settings for all tasks."""

    # Prompt with LoRA reminder
    prompt_text = "Enter your prompt"
    if CURRENT_LORA_PATH:
        lora_name = os.path.basename(CURRENT_LORA_PATH)
        prompt_text += f" (Active LoRA: [cyan]{lora_name}[/cyan])"

    prompt = Prompt.ask(prompt_text, default="A beautiful digital art of a sunset")
    neg_prompt = Prompt.ask("Enter negative prompt", default="")
    if not neg_prompt: neg_prompt = None

    schedulers = utils.get_scheduler_list()
    console.print("[bold]Select Sampler:[/bold]")
    for idx, s in enumerate(schedulers):
        console.print(f"{idx+1}. {s}")
    sched_choice = IntPrompt.ask("Choose sampler", default=1, choices=[str(i+1) for i in range(len(schedulers))])
    scheduler_name = schedulers[sched_choice-1]

    steps = IntPrompt.ask("Inference steps", default=15)
    cfg_scale = FloatPrompt.ask("CFG Scale", default=7.0)
    seed = IntPrompt.ask("Seed (-1 for random)", default=-1)
    batch_size = IntPrompt.ask("Batch Size", default=1)
    batch_count = IntPrompt.ask("Batch Count", default=1)

    return {
        "prompt": prompt, "negative_prompt": neg_prompt, "scheduler": scheduler_name,
        "steps": steps, "guidance_scale": cfg_scale, "seed": seed,
        "batch_size": batch_size, "batch_count": batch_count
    }

def select_lora_menu():
    """Menu to select current LoRA."""
    global CURRENT_LORA_PATH, CURRENT_LORA_SCALE

    loras = lora_manager.list_loras()

    console.print("[bold]Select LoRA:[/bold]")
    console.print("0. None (Disable LoRA)")
    console.print("1. Enter Manual Path / HF ID")

    for i, l in enumerate(loras):
        console.print(f"{i+2}. {l['name']}")

    choice = Prompt.ask("Choose LoRA", default="0", choices=[str(i) for i in range(len(loras)+2)])

    if choice == "0":
        CURRENT_LORA_PATH = None
        console.print("[yellow]LoRA Disabled[/yellow]")
    elif choice == "1":
        CURRENT_LORA_PATH = Prompt.ask("Enter path or HF ID")
    else:
        CURRENT_LORA_PATH = loras[int(choice)-2]["path"]

    if CURRENT_LORA_PATH:
        CURRENT_LORA_SCALE = FloatPrompt.ask("LoRA Scale", default=1.0)
        console.print(f"[green]Selected LoRA: {os.path.basename(CURRENT_LORA_PATH)} ({CURRENT_LORA_SCALE})[/green]")

    # Invalidate cache
    global CACHED_PIPELINE
    CACHED_PIPELINE = None

def select_model():
    """Menu to select current model."""
    global CURRENT_MODEL_PATH, CURRENT_MODEL_TYPE

    local_models = model_manager.list_models()

    table = Table(title="Select Model")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Type")

    options = []
    options.append({"name": "segmind/tiny-sd (Default HF)", "path": "segmind/tiny-sd", "type": "hf_id"})

    for m in local_models:
        options.append(m)

    for i, opt in enumerate(options):
        table.add_row(str(i+1), opt["name"], opt["type"])

    console.print(table)
    console.print(f"D. Download New Model")

    choice = Prompt.ask("Select model", default="1")

    if choice.lower() == "d":
        model_manager.download_menu()
        return select_model()

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(options):
            sel = options[idx]
            CURRENT_MODEL_PATH = sel["path"]
            CURRENT_MODEL_TYPE = sel["type"]
            console.print(f"[green]Selected: {sel['name']}[/green]")
            global CACHED_PIPELINE
            CACHED_PIPELINE = None
        else:
            console.print("[red]Invalid selection[/red]")
    except:
        console.print("[red]Invalid input[/red]")

def run_task(task_name, device_key):
    # Map selection to internal device key
    if device_key == "1": selected_device = "cpu"
    elif device_key == "2": selected_device = "intel"
    elif device_key == "3": selected_device = "openvino_cpu"
    elif device_key == "4": selected_device = "openvino_gpu"
    else: selected_device = device_key

    lora_status = f" | LoRA: {os.path.basename(CURRENT_LORA_PATH)}" if CURRENT_LORA_PATH else ""
    console.print(Panel(f"[bold cyan]{task_name}[/bold cyan] Mode (Running on [bold green]{selected_device.upper()}[/bold green])\nModel: [bold yellow]{CURRENT_MODEL_PATH}[/bold yellow]{lora_status}"))

    if task_name == "Text to Image":
        inputs = get_common_settings()
        width = IntPrompt.ask("Width", default=512)
        height = IntPrompt.ask("Height", default=512)
        output_base = Prompt.ask("Output filename (base)", default="output_txt2img")
        
        pipe = get_pipeline("txt2img", selected_device, width, height, inputs["batch_size"], inputs["scheduler"])
        if not pipe: return

        for i in range(inputs["batch_count"]):
            current_seed = inputs["seed"] if inputs["seed"] != -1 else torch.randint(0, 2**32, (1,)).item()
            generator = torch.manual_seed(current_seed)
            console.print(f"[dim]Batch {i+1}/{inputs['batch_count']} | Seed: {current_seed}[/dim]")

            with Progress(SpinnerColumn(), TextColumn("Generating..."), transient=True) as progress:
                progress.add_task("gen", total=None)
                # Pass scale if PyTorch
                kwargs = {}
                if "openvino" not in selected_device and CURRENT_LORA_PATH:
                    kwargs["cross_attention_kwargs"] = {"scale": CURRENT_LORA_SCALE}

                images = pipe(
                    prompt=inputs["prompt"], negative_prompt=inputs["negative_prompt"],
                    num_inference_steps=inputs["steps"], guidance_scale=inputs["guidance_scale"],
                    width=width, height=height, num_images_per_prompt=inputs["batch_size"], generator=generator,
                    **kwargs
                ).images

            for j, img in enumerate(images):
                suffix = f"_{i}_{j}.png" if (inputs["batch_count"] > 1 or inputs["batch_size"] > 1) else ".png"
                img.save(f"{output_base}{suffix}")
        console.print(f"[bold green]Success![/bold green]")

    elif task_name in ["Image to Image", "Inpainting", "Outpainting", "Variations"]:
        img_path = Prompt.ask("Enter path to input image")
        init_image = load_image(img_path)
        if not init_image: return

        orig_w, orig_h = init_image.size
        width = (orig_w // 64) * 64
        height = (orig_h // 64) * 64

        if task_name == "Outpainting":
            padding = IntPrompt.ask("Padding pixels", default=128)
            width = ((orig_w + padding*2) // 64) * 64
            height = ((orig_h + padding*2) // 64) * 64

        inputs = get_common_settings()
        strength = FloatPrompt.ask("Strength", default=0.75)
        output_base = Prompt.ask("Output filename (base)", default="output_img")

        pipe = get_pipeline("img2img", selected_device, width, height, inputs["batch_size"], inputs["scheduler"])
        if not pipe: return

        if task_name == "Outpainting":
            padded_image = ImageOps.expand(init_image, border=padding, fill="gray")
            mask_image = Image.new("L", padded_image.size, 255)
            mask_image.paste(0, (padding, padding, padding + orig_w, padding + orig_h))
            input_image = padded_image.resize((width, height))
        elif task_name == "Inpainting":
            mask_path_in = Prompt.ask("Enter path to mask image")
            mask_image_raw = load_image(mask_path_in)
            if not mask_image_raw: return
            init_image = init_image.resize((width, height))
            mask_image = mask_image_raw.convert("L").resize((width, height))
            input_image = init_image
        else:
            input_image = init_image.resize((width, height))

        for i in range(inputs["batch_count"]):
            current_seed = inputs["seed"] if inputs["seed"] != -1 else torch.randint(0, 2**32, (1,)).item()
            generator = torch.manual_seed(current_seed)

            with Progress(SpinnerColumn(), TextColumn("Processing..."), transient=True) as progress:
                progress.add_task("gen", total=None)
                prompt = "" if task_name == "Variations" else inputs["prompt"]

                kwargs = {}
                if "openvino" not in selected_device and CURRENT_LORA_PATH:
                    kwargs["cross_attention_kwargs"] = {"scale": CURRENT_LORA_SCALE}

                gens = pipe(
                    prompt=prompt, negative_prompt=inputs["negative_prompt"],
                    image=input_image, strength=strength,
                    num_inference_steps=inputs["steps"], guidance_scale=inputs["guidance_scale"],
                    num_images_per_prompt=inputs["batch_size"], generator=generator,
                    **kwargs
                ).images

                for j, gen in enumerate(gens):
                    if task_name == "Outpainting":
                        gen = gen.resize(padded_image.size)
                        final = Image.composite(gen, padded_image, mask_image)
                    elif task_name == "Inpainting":
                        final = Image.composite(gen, init_image, mask_image)
                    else:
                        final = gen

                    suffix = f"_{i}_{j}.png" if (inputs["batch_count"] > 1 or inputs["batch_size"] > 1) else ".png"
                    final.save(f"{output_base}{suffix}")
        console.print(f"[bold green]Success![/bold green]")

def main():
    console.print(Panel.fit(
        "[bold cyan]Tiny-SD Studio Pro[/bold cyan]\n[italic white]Optimized for Intel OpenVINO, CPU & HD 620[/italic white]",
        border_style="magenta"
    ))
    
    # Session Setup
    device_table = Table(title="1. Select Hardware")
    device_table.add_column("Key", style="cyan")
    device_table.add_column("Hardware")
    device_table.add_row("1", "CPU (Standard)")
    device_table.add_row("2", "Intel GPU (DirectML)")
    device_table.add_row("3", "Intel OpenVINO CPU")
    device_table.add_row("4", "Intel OpenVINO GPU")
    console.print(device_table)
    hw_choice = Prompt.ask("Choose hardware", choices=["1", "2", "3", "4"], default="3")

    select_model()
    
    while True:
        lora_status = f" (LoRA: {os.path.basename(CURRENT_LORA_PATH)})" if CURRENT_LORA_PATH else ""
        table = Table(title="[bold magenta]Main Menu[/bold magenta]")
        table.add_column("Option", width=6)
        table.add_column("Task")
        table.add_row("1", f"Text to Image{lora_status}")
        table.add_row("2", f"Image to Image{lora_status}")
        table.add_row("3", f"Inpainting{lora_status}")
        table.add_row("4", f"Outpainting{lora_status}")
        table.add_row("5", f"Variations{lora_status}")
        table.add_row("M", "Change Model")
        table.add_row("L", "Select LoRA (Optional)")
        table.add_row("D", "Download LoRA")
        table.add_row("H", "Change Hardware")
        table.add_row("0", "Exit")
        console.print(table)
        
        choice = Prompt.ask("Select option", choices=["0", "1", "2", "3", "4", "5", "M", "L", "D", "H"], default="1")
        
        if choice == "0":
            break
        elif choice == "M":
            select_model()
        elif choice == "L":
            select_lora_menu()
        elif choice == "D":
            lora_manager.download_menu()
        elif choice == "H":
            hw_choice = Prompt.ask("Choose hardware", choices=["1", "2", "3", "4"])
            global CACHED_PIPELINE
            CACHED_PIPELINE = None
        else:
            tasks = {
                "1": "Text to Image", "2": "Image to Image", "3": "Inpainting",
                "4": "Outpainting", "5": "Variations"
            }
            run_task(tasks[choice], hw_choice)
        
        console.print("\n" + "="*40 + "\n")

if __name__ == "__main__":
    main()
