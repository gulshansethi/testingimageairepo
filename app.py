# Command to start
# streamlit run app.py OR streamlit run app.py --server.port 8502


import os
import streamlit as st
from datetime import datetime
from image_generator import generate_image, IS_SDXL

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="centered",
)

st.title("🎨 AI Image Generator")
st.caption("Powered by Stable Diffusion · Runs 100% on your machine · No API required")

# ── Sidebar (settings) ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    quality_preset = st.selectbox(
        "Speed preset",
        ["Fast", "Balanced", "High Quality"],
        index=0,
        help="Fast is best when your machine is slow.",
    )

    preset_steps = {"Fast": 10, "Balanced": 16, "High Quality": 28}
    preset_guidance = {"Fast": 6.0, "Balanced": 7.0, "High Quality": 8.0}

    num_steps = st.slider(
        "Inference Steps",
        min_value=4,
        max_value=50,
        value=preset_steps[quality_preset],
        help="More steps = higher quality, but slower generation.",
    )

    guidance_scale = st.slider(
        "Guidance Scale",
        min_value=1.0,
        max_value=15.0,
        value=preset_guidance[quality_preset],
        step=0.5,
        help="How strictly the model follows your prompt. 7-8 is a good default.",
    )

    col1, col2 = st.columns(2)
    size_options = [512, 640, 768] if IS_SDXL else [384, 512, 640]
    default_size = 512
    with col1:
        width = st.selectbox("Width", size_options, index=size_options.index(default_size))
    with col2:
        height = st.selectbox("Height", size_options, index=size_options.index(default_size))

    use_seed = st.checkbox("Fix random seed (reproducible)")
    seed = None
    if use_seed:
        seed = st.number_input("Seed", min_value=0, max_value=2**32 - 1, value=42)

    st.divider()
    save_images = st.checkbox("Auto-save generated images", value=True)

    st.divider()
    st.info(
        "**Tip:** Keep 512x512 and Fast/Balanced preset for quicker generation. "
        "First run downloads model files once and caches them locally. "
        "Subsequent runs are instant."
    )

# ── Main area ────────────────────────────────────────────────────────────────
prompt = st.text_area(
    "✏️ Describe the image you want",
    placeholder="e.g. A futuristic city at night with neon lights, cinematic, 4K",
    height=100,
)

negative_prompt = st.text_input(
    "🚫 Negative prompt  (optional — things to avoid)",
    placeholder="e.g. blurry, distorted, low quality, watermark",
)

generate_btn = st.button("🚀 Generate Image", type="primary", use_container_width=True)

# ── Generation logic ─────────────────────────────────────────────────────────
if generate_btn:
    if not prompt.strip():
        st.warning("Please enter a prompt before generating.")
    else:
        with st.spinner("Generating image… (first run may take a minute to load the model)"):
            try:
                image = generate_image(
                    prompt=prompt.strip(),
                    negative_prompt=negative_prompt.strip() if negative_prompt.strip() else None,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    seed=int(seed) if use_seed else None,
                )

                st.image(image, caption=prompt, use_column_width=True)

                # Save to disk
                filename = None
                if save_images:
                    os.makedirs("generated_images", exist_ok=True)
                    filename = os.path.join(
                        "generated_images",
                        f"img_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    )
                    image.save(filename)
                    st.success(f"Image saved → `{filename}`")

                # Download button (always available)
                from io import BytesIO
                buf = BytesIO()
                image.save(buf, format="PNG")
                st.download_button(
                    label="⬇️ Download Image",
                    data=buf.getvalue(),
                    file_name=f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True,
                )

            except Exception as e:
                st.error(f"Generation failed: {e}")
                st.exception(e)
