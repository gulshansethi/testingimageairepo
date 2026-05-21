# Command to start
# streamlit run app.py OR streamlit run app.py --server.port 8502

import os

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from PIL import Image
import streamlit as st

# Lightweight default model for faster local execution
MODEL_ID = os.getenv("MODEL_ID", "runwayml/stable-diffusion-v1-5")
IS_SDXL = False


@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load a single lightweight pipeline for faster local execution."""
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

    print(f"[INFO] Loading SD model on {device} (dtype={dtype}) ...")

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

    pipe.set_progress_bar_config(disable=True)

    print("[INFO] Model loaded successfully.")
    return pipe


def generate_image(
    prompt: str,
    negative_prompt: str = None,
    num_inference_steps: int = 8,
    guidance_scale: float = 6.0,
    width: int = 384,
    height: int = 384,
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
