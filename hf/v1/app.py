import gradio as gr
import numpy as np
import random
import spaces
import torch
from diffusers import DiffusionPipeline
import gc

# Define models and their configurations
models = {
    "FLUX.1-dev": {
        "pipeline_class": DiffusionPipeline,
        "model_id": "black-forest-labs/FLUX.1-dev",
        "config": {"torch_dtype": torch.bfloat16},
        "description": "**FLUX.1-dev** is a development model that focuses on delivering highly detailed and artistically rich images.",
        "status": "Not loaded"
    },
    "FLUX.1-schnell": {
        "pipeline_class": DiffusionPipeline,
        "model_id": "black-forest-labs/FLUX.1-schnell",
        "config": {"torch_dtype": torch.bfloat16},
        "description": "**FLUX.1-schnell** is a rectified flow transformer model optimized for fast 4-step generation, distilled from the [FLUX.1-pro](https://blackforestlabs.ai/) model for quick and efficient image synthesis.",
        "status": "Not loaded"
    },
}

dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize variables
selected_model_key = "FLUX.1-schnell"
selected_model = models[selected_model_key]
pipe = None
loaded_model_key = None  # Keeps track of the currently loaded model

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048

def clean_gpu():
    """Frees GPU memory by deleting the pipeline, clearing the cache, and collecting garbage."""
    global pipe
    if pipe is not None:
        del pipe
        pipe = None  # Explicitly set pipe to None
    torch.cuda.empty_cache()
    gc.collect()  # Collect garbage to further free up memory

def load_model(model_key):
    """Loads the specified model and updates its status."""
    global pipe, selected_model, loaded_model_key
    if model_key == loaded_model_key:
        return  # Model already loaded, no need to reload

    clean_gpu()  # Clean up before loading the new model
    selected_model = models[model_key]
    selected_model["status"] = "Loading..."
    try:
        pipe = selected_model["pipeline_class"].from_pretrained(
            selected_model["model_id"], **selected_model["config"]
        ).to(device)
        selected_model["status"] = "Loaded"
        loaded_model_key = model_key
    except Exception as e:
        selected_model["status"] = f"Error: {e}"  # Provide error information
        loaded_model_key = None

# Load the default model initially
load_model(selected_model_key)

@spaces.GPU()
def infer(model_key, prompt, seed=42, randomize_seed=False, width=1024, height=1024, num_inference_steps=4, progress=gr.Progress(track_tqdm=True)):
    global loaded_model_key
    if model_key != loaded_model_key or models[model_key]["status"] != "Loaded":
        return "Please load the model before generating images.", None

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(seed)
    image = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        generator=generator,
        guidance_scale=0.0
    ).images[0]
    return image, seed

examples = [
    ["FLUX.1-schnell", "a tiny astronaut hatching from an egg on the moon"],
    ["FLUX.1-schnell", "a cat holding a sign that says hello world"],
    ["FLUX.1-schnell", "an anime illustration of a wiener schnitzel"],
]

css="""
#col-container {
    margin: 0 auto;
    max-width: 520px;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""# FLUX Image Generator
Select from multiple models to generate images. Each model is designed for different use cases.
""")

        model_selector = gr.Dropdown(
            label="Select Model",
            choices=list(models.keys()),
            value=selected_model_key,
            interactive=True
        )

        description_box = gr.Markdown(f"""{selected_model["description"]}""")
        status_box = gr.Textbox(value=models[selected_model_key]["status"], label="Model Status", interactive=False)

        def update_description_and_status(model_key):
            return models[model_key]["description"], models[model_key]["status"]

        model_selector.change(
            fn=update_description_and_status,
            inputs=model_selector,
            outputs=[description_box, status_box]
        )

        load_button = gr.Button("Load Model")

        def load_selected_model(model_key):
            load_model(model_key)
            return models[model_key]["status"]

        load_button.click(
            fn=load_selected_model,
            inputs=model_selector,
            outputs=status_box
        )

        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )

            run_button = gr.Button("Run", scale=0, interactive=(selected_model["status"] == "Loaded"))

        def enable_run_button(status):
            return gr.update(interactive=(status == "Loaded"))

        status_box.change(
            fn=enable_run_button,
            inputs=status_box,
            outputs=run_button
        )

        result = gr.Image(label="Result", show_label=False)

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
                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=4,
                )

        gr.Examples(
            examples=examples,
            fn=infer,
            inputs=[model_selector, prompt],
            outputs=[result, seed],
            cache_examples=True,
            cache_mode="lazy"
        )

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[model_selector, prompt, seed, randomize_seed, width, height, num_inference_steps],
        outputs=[result, seed]
    )

demo.launch()
