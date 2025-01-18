import os
import torch
import gradio as gr
from diffusers import FluxPipeline, DiffusionPipeline

# Helper function to get the Hugging Face token securely
def get_hf_token():
    try:
        from google.colab import userdata
        hf_token = userdata.get('HF_TOKEN')
        if hf_token:
            return hf_token
        else:
            raise RuntimeError("HF_TOKEN not found in Colab secrets.")
    except ImportError:
        return os.getenv("HF_TOKEN", None)

# Securely get the token
_HF_TOKEN = get_hf_token()
if not _HF_TOKEN:
    raise ValueError("HF_TOKEN is not available. Please set it in Colab secrets or environment variables.")

# Define models and their configurations
models = {
    "FLUX.1-schnell": {
        "pipeline_class": FluxPipeline,
        "model_id": "black-forest-labs/FLUX.1-schnell",
        "config": {"torch_dtype": torch.bfloat16},
        "description": "**FLUX.1-schnell** is a fast and efficient model designed for quick image generation. It excels at producing high-quality images rapidly, making it ideal for applications where speed is crucial. However, its rapid generation may slightly compromise on the level of detail compared to slower, more meticulous models.",
    },
    "FLUX.1-dev": {
        "pipeline_class": DiffusionPipeline,
        "model_id": "black-forest-labs/FLUX.1-dev",
        "lora": {
            "repo": "strangerzonehf/Flux-Enrich-Art-LoRA",
            "trigger_word": "enrich art",
        },
        "config": {"torch_dtype": torch.bfloat16},
        "description": "**FLUX.1-dev** is a development model that focuses on delivering highly detailed and artistically rich images.",
    },
    "Flux.1-lite-8B-alpha": {
        "pipeline_class": FluxPipeline,
        "model_id": "Freepik/flux.1-lite-8B-alpha",
        "config": {"torch_dtype": torch.bfloat16},
        "description": "**Flux.1-lite-8B-alpha** is a lightweight model optimized for efficiency and ease of use.",
    },
}

# Function to pre-download models
def download_all_models():
    print("Downloading all models...")
    for model_key, config in models.items():
        try:
            pipeline_class = config["pipeline_class"]
            model_id = config["model_id"]
            # Attempt to download the pipeline without loading it into memory
            pipeline_class.from_pretrained(model_id, token=_HF_TOKEN, **config.get("config", {}))
            if "lora" in config:
                pipeline_class.download_lora_weights(config["lora"]["repo"],token=_HF_TOKEN,)
            print(f"Model '{model_key}' downloaded successfully.")
        except Exception as e:
            print(f"Error downloading model '{model_key}': {e}")
    print("Model download process complete.")

loaded_models = {}
model_load_status = {}  # Dictionary to track model load status

def clear_gpu_memory():
    """Clears GPU memory. Keeps model status information."""
    global loaded_models
    try:
        for model_key in list(loaded_models.keys()):  # Iterate over a copy to allow deletion
            del loaded_models[model_key]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print("GPU memory cleared.")
        return "GPU memory cleared."
    except Exception as e:
        print(f"Error clearing GPU memory: {e}")
        return f"Error clearing GPU memory: {e}"


def load_model(model_key):
    """Loads a model, clearing GPU memory first if a different model was loaded."""
    global model_load_status

    if model_key not in models:
        model_load_status[model_key] = "Model not found."
        return f"Model '{model_key}' not found in the available models."

    # Clear GPU memory only if a different model is already loaded
    if loaded_models and list(loaded_models.keys())[0] != model_key:
        clear_gpu_memory()

    try:
        config = models[model_key]
        pipeline_class = config["pipeline_class"]
        model_id = config["model_id"]
        
        pipe = pipeline_class.from_pretrained(model_id, token=_HF_TOKEN, **config.get("config", {}))

        if "lora" in config:
            lora_config = config["lora"]
            pipe.load_lora_weights(lora_config["repo"], token=_HF_TOKEN)

        if torch.cuda.is_available():
            pipe.to("cuda")

        loaded_models[model_key] = pipe
        model_load_status[model_key] = "Loaded" # Update load status
        return f"Model '{model_key}' loaded successfully."
    except Exception as e:
        model_load_status[model_key] = "Failed" # Update load status on error
        return f"Error loading model '{model_key}': {e}"

def generate_image(model, prompt, seed=-1):
    
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    if seed != -1:
        generator = generator.manual_seed(seed)

    with torch.no_grad():
        image = model(prompt=prompt, generator=generator).images[0]
    return image

def gradio_generate(selected_model, prompt, seed):
    
    if selected_model not in loaded_models:
      if selected_model in model_load_status and model_load_status[selected_model] == "Loaded":
          # Model should be loaded but isn't in loaded_models, clear it from the status
          del model_load_status[selected_model]
      if selected_model not in model_load_status or model_load_status[selected_model] != "Loaded":
          # Attempt to load the model if not already attempted or failed
          load_model(selected_model)
      
      if selected_model not in loaded_models:
          # If still not loaded after attempt, return an error
          return f"Model not loaded. Load status: {model_load_status.get(selected_model, 'Not attempted')}.", None

    model = loaded_models[selected_model]
    image = generate_image(model, prompt, seed)

    runtime_info = f"Model: {selected_model}\nSeed: {seed}"
    output_path = "generated_image.png"
    image.save(output_path)
    return output_path, runtime_info

def gradio_load_model(selected_model):
    if not selected_model:
        return "No model selected. Please select a model to load."
    return load_model(selected_model)



import gradio as gr

with gr.Blocks(
    css="""
    .container {
        max-width: 800px;
        margin: auto;
        padding: 20px;
        background-color: #ffffff; /* Pure white for a clean background */
        border-radius: 10px; /* Smooth rounded corners */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05); /* Subtle shadow for a lighter feel */
        color: #333333; /* Dark gray text for good contrast and readability */
    }



    """
) as interface:
    with gr.Tab("Image Generator"):
        with gr.Column():
            gr.Markdown("# Text-to-Image Generator")
            model_dropdown = gr.Dropdown(choices=list(models.keys()), label="Select Model")
            prompt_textbox = gr.Textbox(label="Enter Text Prompt")
            seed_slider = gr.Slider(minimum=-1, maximum=1000, step=1, value=-1, label="Random Seed (-1 for random)")
            generate_button = gr.Button("Generate Image")
            output_image = gr.Image(label="Generated Image")
            runtime_info_textbox = gr.Textbox(label="Runtime Information", lines=2, interactive=False)

            # Add example prompts at the bottom
            gr.Markdown("### Example Prompts")
            examples = gr.Examples(
                examples=[
                    [list(models.keys())[0], "Sexy Woman", "sample2.png"],
                    [list(models.keys())[2], "Sexy girl", "sample3.png"],
                    [list(models.keys())[1], "Future City", "sample1.png"]
                ],
                inputs=[model_dropdown, prompt_textbox, output_image],
            )

            generate_button.click(gradio_generate, inputs=[model_dropdown, prompt_textbox, seed_slider], outputs=[output_image, runtime_info_textbox])

    with gr.Tab("Model Information"):
        for model_key, model_info in models.items():
            gr.Markdown(f"## {model_key}")
            gr.Markdown(model_info["description"])

    gr.Markdown("""---
    **Credits**: Created by Ruslan Magana Vsevolodovna. For more information, visit [https://ruslanmv.com/](https://ruslanmv.com/).""")

if __name__ == "__main__":

    # Pre-download all models at startup
    download_all_models()  
    interface.launch(debug=True)


