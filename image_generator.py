# Command to start
# streamlit run app.py OR streamlit run app.py --server.port 8502

import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
)
from PIL import Image
import streamlit as st

# Change this to any Stable Diffusion model ID from Hugging Face
# SD v1.x / v2.x examples : "runwayml/stable-diffusion-v1-5"
#                            "stabilityai/stable-diffusion-2-1"
# SDXL examples            : "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# Auto-detect if the model is SDXL based on its name
IS_SDXL = "xl" in MODEL_ID.lower()


@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load the correct pipeline automatically based on MODEL_ID."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Safe speedups for modern PyTorch builds
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print(f"[INFO] Loading {'SDXL' if IS_SDXL else 'SD'} model on {device} (dtype={dtype}) ...")

    if IS_SDXL:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if device == "cuda" else None,
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

    # Faster scheduler with good quality at lower step counts
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to(device)

    # Reduce memory usage on CPU
    if device == "cpu":
        pipe.enable_attention_slicing()
    else:
        # Optional GPU speedup if xformers exists
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    pipe.set_progress_bar_config(disable=True)

    print("[INFO] Model loaded successfully.")
    return pipe


def generate_image(
    prompt: str,
    negative_prompt: str = None,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    width: int = 512,
    height: int = 512,
    seed: int = None,
) -> Image.Image:
    """
    Generate an image from a text prompt using Stable Diffusion locally.

    Args:
        prompt:               What you want the image to look like.
        negative_prompt:      What you DON'T want (e.g. 'blurry, low quality').
        num_inference_steps:  More steps → better quality but slower (10-50).
        guidance_scale:       How strictly to follow the prompt (1-15).
        width / height:       Output image size in pixels (must be multiples of 8).
                              SDXL works best at 1024x1024.
        seed:                 Fixed seed for reproducible results (None = random).

    Returns:
        A PIL Image object.
    """
    pipe = load_pipeline()

    generator = None
    if seed is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
    )

    return result.images[0]
