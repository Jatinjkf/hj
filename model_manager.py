import os
import requests
import re
from rich.console import Console
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, DownloadColumn, TransferSpeedColumn, TextColumn, TimeRemainingColumn
from huggingface_hub import snapshot_download

console = Console()
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def list_models():
    """List available models in the local directory."""
    models = []
    if not os.path.exists(MODEL_DIR):
        return models

    for item in os.listdir(MODEL_DIR):
        path = os.path.join(MODEL_DIR, item)
        if os.path.isdir(path):
            if item not in ["openvino", "pytorch", "openvino_cache"]: # Exclude cache dirs
                models.append({"name": item, "type": "folder", "path": path})
        elif item.endswith(".safetensors") or item.endswith(".ckpt"):
            models.append({"name": item, "type": "file", "path": path})

    return models

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
        # Use stream=True to get headers before downloading body
        with requests.get(url, stream=True, allow_redirects=True) as r:
            r.raise_for_status()

            # Try to guess filename if not provided
            if not filename or filename == "model.safetensors":
                cd = r.headers.get("content-disposition")
                guessed_name = get_filename_from_cd(cd)
                if guessed_name:
                    filename = guessed_name
                    console.print(f"[dim]Detected filename: {filename}[/dim]")
                else:
                    # Fallback to URL
                    if "model.safetensors" in filename: # If default was passed but we want to try URL
                        url_name = url.split("/")[-1].split("?")[0]
                        if "." in url_name:
                            filename = url_name

            # Final check
            if not filename:
                filename = Prompt.ask("Could not detect filename. Enter name to save as (e.g. model.safetensors)")

            path = os.path.join(MODEL_DIR, filename)
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

def download_from_hf(repo_id):
    """Download a model from Hugging Face."""
    console.print(f"Downloading {repo_id} from Hugging Face...")
    try:
        folder_name = repo_id.replace("/", "--")
        local_dir = os.path.join(MODEL_DIR, folder_name)
        snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
        console.print(f"[bold green]Successfully downloaded to {local_dir}[/bold green]")
        return local_dir
    except Exception as e:
        console.print(f"[bold red]Download failed:[/bold red] {e}")
        return None

def download_menu():
    console.print("[bold cyan]Model Downloader[/bold cyan]")
    console.print("1. Hugging Face (Repo ID)")
    console.print("2. URL (Civitai, Direct Link, etc.)")
    console.print("0. Back")

    choice = Prompt.ask("Select source", choices=["1", "2", "0"], default="1")

    if choice == "1":
        repo_id = Prompt.ask("Enter HF Repo ID (e.g. 'segmind/tiny-sd')")
        download_from_hf(repo_id)
    elif choice == "2":
        url = Prompt.ask("Enter Direct URL")
        download_file(url, "model.safetensors") # Pass default, function will auto-detect

if __name__ == "__main__":
    while True:
        download_menu()
        if Prompt.ask("Download another?", choices=["y", "n"], default="n") == "n":
            break
