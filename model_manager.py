import os
import requests
from rich.console import Console
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, DownloadColumn, TransferSpeedColumn, TextColumn, TimeRemainingColumn
from huggingface_hub import snapshot_download

console = Console()
MODEL_DIR = "tiny-sd-models"
os.makedirs(MODEL_DIR, exist_ok=True)

def list_models():
    """List available models in the local directory."""
    models = []
    if not os.path.exists(MODEL_DIR):
        return models

    for item in os.listdir(MODEL_DIR):
        path = os.path.join(MODEL_DIR, item)
        if os.path.isdir(path):
            # Assume diffusers folder or folder containing checkpoints
            if item not in ["openvino", "pytorch", "openvino_cache"]: # Exclude cache dirs
                models.append({"name": item, "type": "folder", "path": path})
        elif item.endswith(".safetensors") or item.endswith(".ckpt"):
            models.append({"name": item, "type": "file", "path": path})

    return models

def download_file(url, filename):
    """Download a file with progress bar."""
    path = os.path.join(MODEL_DIR, filename)

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
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

def download_from_hf(repo_id):
    """Download a model from Hugging Face."""
    console.print(f"Downloading {repo_id} from Hugging Face...")
    try:
        # Check if it's a full diffusers repo or we want a specific file?
        # Usually snapshot_download is safest for diffusers.
        # Save to subdirectory named after repo
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
    console.print("2. URL (Direct Link to .safetensors/.ckpt)")
    console.print("0. Back")

    choice = Prompt.ask("Select source", choices=["1", "2", "0"], default="1")

    if choice == "1":
        repo_id = Prompt.ask("Enter HF Repo ID (e.g. 'segmind/tiny-sd')")
        download_from_hf(repo_id)
    elif choice == "2":
        url = Prompt.ask("Enter Direct URL")
        filename = url.split("/")[-1]
        if "?" in filename: filename = filename.split("?")[0]
        if not (filename.endswith(".safetensors") or filename.endswith(".ckpt")):
            filename = Prompt.ask("Enter filename to save as (e.g. model.safetensors)", default="model.safetensors")
        download_file(url, filename)

if __name__ == "__main__":
    while True:
        download_menu()
        if Prompt.ask("Download another?", choices=["y", "n"], default="n") == "n":
            break
