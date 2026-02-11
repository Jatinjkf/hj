import os
import warnings
from diffusers import (
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler
)

def suppress_warnings():
    """Suppress annoying warnings from huggingface_hub and diffusers."""
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    # Filter specific warnings if needed, or just general ones that clutter the output
    warnings.filterwarnings("ignore", message=".*symlinks.*")
    warnings.filterwarnings("ignore", message=".*unsafe serialization.*")
    # Also ignore FutureWarning about resume_download
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*resume_download.*")

def ensure_extension(filename, ext=".png"):
    """Ensure the filename has an extension."""
    if not filename:
        return f"output{ext}"
    if not os.path.splitext(filename)[1]:
        return f"{filename}{ext}"
    return filename

def get_scheduler_list():
    """Return a list of available schedulers for the menu."""
    return [
        "DPM++ 2M Karras", # Default/Recommended
        "DPM++ 2M",
        "Euler",
        "Euler Ancestral", # Euler a
        "LMS",
        "PNDM",
        "DDIM"
    ]

def get_optimal_scheduler(pipe, name="DPM++ 2M Karras"):
    """Replace the pipeline scheduler based on the selection."""
    config = pipe.scheduler.config

    if name == "DPM++ 2M Karras":
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)
    elif name == "DPM++ 2M":
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=False)
    elif name == "Euler":
        return EulerDiscreteScheduler.from_config(config)
    elif name == "Euler Ancestral":
        return EulerAncestralDiscreteScheduler.from_config(config)
    elif name == "LMS":
        return LMSDiscreteScheduler.from_config(config)
    elif name == "PNDM":
        return PNDMScheduler.from_config(config)
    elif name == "DDIM":
        return DDIMScheduler.from_config(config)
    else:
        # Fallback to DPM++ 2M Karras
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)

def configure_scheduler(pipe, name):
    """Convenience function to set scheduler on pipe."""
    pipe.scheduler = get_optimal_scheduler(pipe, name)
    return pipe
