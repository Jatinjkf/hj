import os
import warnings
from diffusers import DPMSolverMultistepScheduler

def suppress_warnings():
    """Suppress annoying warnings from huggingface_hub and diffusers."""
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    # Filter specific warnings if needed, or just general ones that clutter the output
    warnings.filterwarnings("ignore", message=".*symlinks.*")
    warnings.filterwarnings("ignore", message=".*unsafe serialization.*")

def ensure_extension(filename, ext=".png"):
    """Ensure the filename has an extension."""
    if not filename:
        return f"output{ext}"
    if not os.path.splitext(filename)[1]:
        return f"{filename}{ext}"
    return filename

def get_optimal_scheduler(pipe):
    """Replace the default scheduler with DPMSolverMultistepScheduler for speed."""
    # use_karras_sigmas=True is often better for quality at low steps
    return DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
