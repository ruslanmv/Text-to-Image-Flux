import gradio as gr
import numpy as np
import random
import spaces
import torch
from diffusers import DiffusionPipeline, AutoencoderTiny, AutoencoderKL, StableDiffusionPipeline
from live_preview_helpers import flux_pipe_call_that_returns_an_iterable_of_images
from PIL import Image
import io
import zipfile
from huggingface_hub import HfApi
_HF_TOKEN = HfApi().token

# --- Constants ---
DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048
# --- Model Definitions ---
MODELS  = {
    "FLUX.1-schnell": {
        "pipeline_class": FluxPipeline,
        "model_id": "black-forest-labs/FLUX.1-schnell",
        "config": {"torch_dtype": torch.bfloat16},
        "description": "**FLUX.1-schnell** is a fast and efficient model designed for quick image generation. It excels at producing high-quality images rapidly, making it ideal for applications where speed is crucial. However, its rapid generation may slightly compromise on the level of detail compared to slower, more meticulous models.",
    },
}
# --- Function to pre-download models ---
def download_all_models():
    print("Downloading all models...")
    for model_key, config in MODELS.items():
        try:
            pipeline_class = config["pipeline_class"]
            model_id = config["model_id"]
            # Attempt to download the pipeline without loading it into memory
            pipeline_class.download(model_id, token=_HF_TOKEN, **config.get("config", {}))

            print(f"Model '{model_key}' downloaded successfully.")
        except Exception as e:
            print(f"Error downloading model '{model_key}': {e}")
    print("Model download process complete.")

# --- Function to clear GPU memory ---
def clear_gpu_memory():
    if DEVICE == "cuda":
        with torch.no_grad():
            torch.cuda.empty_cache()

# --- Function to load models and setup pipeline ---
def load_models(model_key):
    clear_gpu_memory()
    model_info = MODELS[model_key]
    pipeline_class = model_info["pipeline_class"]

    if "vae_id" in model_info:
      vae = AutoencoderTiny.from_pretrained(model_info["vae_id"], torch_dtype=DTYPE).to(DEVICE)
      good_vae = AutoencoderKL.from_pretrained(model_info["model_id"], subfolder="vae", torch_dtype=DTYPE).to(DEVICE)
      pipe = pipeline_class.from_pretrained(model_info["model_id"], torch_dtype=DTYPE, vae=vae, **model_info.get("config", {})).to(DEVICE)
      pipe.flux_pipe_call_that_returns_an_iterable_of_images = flux_pipe_call_that_returns_an_iterable_of_images.__get__(pipe)
      return pipe, good_vae

    else:
      pipe = pipeline_class.from_pretrained(model_info["model_id"], torch_dtype=DTYPE, **model_info.get("config", {})).to(DEVICE)
      return pipe, None

# --- Initial model load ---
current_model_key = "FLUX.1-dev"  # Start with FLUX.1-dev
pipe, good_vae = load_models(current_model_key)

# --- Inference function ---
@spaces.GPU(duration=75)
def infer(model_key, prompt, seed=42, randomize_seed=False, width=1024, height=1024, guidance_scale=3.5, num_inference_steps=28, progress=gr.Progress(track_tqdm=True)):
    global pipe, good_vae, current_model_key
    if model_key != current_model_key:
        pipe, good_vae = load_models(model_key)
        current_model_key = model_key

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(seed)

    images = []

    if current_model_key == "FLUX.1-dev":
        for img in pipe.flux_pipe_call_that_returns_an_iterable_of_images(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator,
            output_type="pil",
            good_vae=good_vae,
        ):
            images.append(img)
            yield img, seed, None
    else:
        result  = pipe(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator,
            
        )
        images.extend(result.images)

        for img in result.images:
          yield img, seed, None

    if images:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for i, img in enumerate(images):
                img_buffer = io.BytesIO()
                img.save(img_buffer, format="PNG")
                zf.writestr(f"image_{i}.png", img_buffer.getvalue())
        yield images[-1], seed, zip_buffer
    else:
        yield None, seed, None

# --- Example prompts ---
examples = [
    "a tiny astronaut hatching from an egg on the moon",
    "a cat holding a sign that says hello world",
    "an anime illustration of a wiener schnitzel",
]

# --- CSS for styling ---
css = """
#col-container {
    margin-left: auto;
    margin-right: auto;
    text-align: center;
}
.text-center {
    text-align: center;
}
.title {
    font-size: 1.5rem;
    font-weight: bold;
    margin-bottom: 1rem;
}
.footer {
    text-align: center;
    margin-top: 1rem;
}
.description-text {
    text-align: left;
    margin-bottom: 1rem;
    
}
"""

# --- Gradio Interface ---
with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(
            """
            <div class="title">
            🖼️ AI Image Generator 🖼️
            </div>
            <div class="text-center">
            Choose a model and generate stunning images with AI!
            </div>
            """,
        )
        
        with gr.Tab("Generator"):
            with gr.Row():
                model_selector = gr.Dropdown(
                    label="Select Model",
                    choices=list(MODELS.keys()),
                    value=current_model_key,
                )

            with gr.Row():
                prompt = gr.Text(
                    label="Prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                    container=False,
                )
                run_button = gr.Button("Run", scale=0)
            result = gr.Image(label="Result", show_label=False, elem_id="result-image")
            download_button = gr.Button("Download Results")
            with gr.Accordion("Advanced Settings", open=False):
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=0,
                )
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                with gr.Row():
                    width = gr.Slider(
                        label="Width",
                        minimum=256,
                        maximum=MAX_IMAGE_SIZE,
                        step=32,
                        value=1024,
                    )
                    height = gr.Slider(
                        label="Height",
                        minimum=256,
                        maximum=MAX_IMAGE_SIZE,
                        step=32,
                        value=1024,
                    )
                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=1,
                        maximum=15,
                        step=0.1,
                        value=3.5,
                    )
                    num_inference_steps = gr.Slider(
                        label="Number of inference steps",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=28,
                    )
            gr.Examples(
                examples=examples,
                fn=infer,
                inputs=[model_selector, prompt],
                outputs=[result, seed, download_button],
                cache_examples="lazy",
            )
        with gr.Tab("Model Descriptions"):
            for model_key, model_info in MODELS.items():
                with gr.Accordion(model_key, open=False):
                    gr.Markdown(model_info["description"], elem_classes="description-text")

        gr.Markdown(
            """
            <div class="footer">
            <p>
            ⚡ Powered by <a href="https://www.gradio.app/" target="_blank">Gradio</a> and <a href="https://huggingface.co/spaces" target="_blank">🤗 Spaces</a>.
            </p>
            </div>
            """,
        )

    # --- Event handlers ---
    inputs = [model_selector, prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps]
    outputs = [result, seed, download_button]
    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=inputs,
        outputs=outputs,
    )
    download_event = download_button.click(lambda x: x, inputs=download_button, outputs=download_button, queue=False)

# --- Pre-download all models at startup ---
download_all_models()

# --- Launch the demo ---
demo.queue().launch(debug=True)