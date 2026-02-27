import os
import requests
import re
from rich.console import Console
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, DownloadColumn, TransferSpeedColumn, TextColumn, TimeRemainingColumn
from huggingface_hub import hf_hub_download

console = Console()
LORA_DIR = os.path.join("models", "loras")
os.makedirs(LORA_DIR, exist_ok=True)

def list_loras():
    """List available LoRAs in the local directory."""
    loras = []
    if not os.path.exists(LORA_DIR):
        return loras

    for item in os.listdir(LORA_DIR):
        if item.endswith(".safetensors"):
            path = os.path.join(LORA_DIR, item)
            loras.append({"name": item, "path": path})

    return loras

def get_filename_from_cd(cd):
    """Get filename from content-disposition."""
    if not cd:
        return None
    fname = re.findall('filename="?([^"]+)"?', cd)
    if len(fname) == 0:
        return None
    return fname[0]

def download_file(url, filename=None):
    """Download a file with progress bar and auto-naming."""

    try:
        with requests.get(url, stream=True, allow_redirects=True) as r:
            r.raise_for_status()

            # Try to guess filename if not provided
            if not filename or filename == "lora.safetensors":
                cd = r.headers.get("content-disposition")
                guessed_name = get_filename_from_cd(cd)
                if guessed_name:
                    filename = guessed_name
                    console.print(f"[dim]Detected filename: {filename}[/dim]")
                else:
                    if "lora.safetensors" in filename:
                        url_name = url.split("/")[-1].split("?")[0]
                        if "." in url_name:
                            filename = url_name

            if not filename:
                filename = Prompt.ask("Enter filename to save as (e.g. style.safetensors)")

            path = os.path.join(LORA_DIR, filename)
            total_size = int(r.headers.get('content-length', 0))

            with Progress(
                TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
                SpinnerColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                "•",
                DownloadColumn(),
                "•",
                TransferSpeedColumn(),
                "•",
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task("Downloading...", filename=filename, total=total_size)

                with open(path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        progress.update(task, advance=len(chunk))

        console.print(f"[bold green]Downloaded {filename} to {path}[/bold green]")
        return path
    except Exception as e:
        console.print(f"[bold red]Download failed:[/bold red] {e}")
        return None

def download_from_hf(repo_id, filename):
    """Download a specific LoRA file from Hugging Face."""
    console.print(f"Downloading {filename} from {repo_id}...")
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=LORA_DIR, local_dir_use_symlinks=False)
        console.print(f"[bold green]Successfully downloaded to {path}[/bold green]")
        return path
    except Exception as e:
        console.print(f"[bold red]Download failed:[/bold red] {e}")
        return None

def download_menu():
    console.print("[bold cyan]LoRA Downloader[/bold cyan]")
    console.print("1. Hugging Face (Repo ID + Filename)")
    console.print("2. URL (Civitai, Direct Link, etc.)")
    console.print("0. Back")

    choice = Prompt.ask("Select source", choices=["1", "2", "0"], default="1")

    if choice == "1":
        repo_id = Prompt.ask("Enter HF Repo ID (e.g. 'segmind/tiny-sd-lora')")
        filename = Prompt.ask("Enter filename (e.g. 'pytorch_lora_weights.safetensors')")
        download_from_hf(repo_id, filename)
    elif choice == "2":
        url = Prompt.ask("Enter Direct URL")
        download_file(url, "lora.safetensors")

if __name__ == "__main__":
    while True:
        download_menu()
        if Prompt.ask("Download another?", choices=["y", "n"], default="n") == "n":
            break
