import os
from diffusers import DiffusionPipeline, AutoencoderKL, AutoencoderTiny
import torch

# Define models and their configurations
models = {
    "FLUX.1-dev": {
        "pipeline_class": DiffusionPipeline,
        "model_id": "black-forest-labs/FLUX.1-dev",
        "config": {"torch_dtype": torch.bfloat16},
        "description": "**FLUX.1-dev** is a development model that focuses on delivering highly detailed and artistically rich images.",
    },
    "FLUX.1-schnell": {
        "pipeline_class": DiffusionPipeline,
        "model_id": "black-forest-labs/FLUX.1-schnell",
        "config": {"torch_dtype": torch.bfloat16},
        "description": "**FLUX.1-schnell** is a rectified flow transformer model optimized for fast 4-step generation, distilled from the [FLUX.1-pro](https://blackforestlabs.ai/) model for quick and efficient image synthesis.",
    },
}

# Helper function to get the Hugging Face token securely
def get_hf_token():
    try:
        from google.colab import userdata  # Try to get token from Colab secrets
        hf_token = userdata.get("HF_TOKEN")
        if hf_token:
            return hf_token
        else:
            raise RuntimeError("HF_TOKEN not found in Colab secrets.")
    except ImportError:  # Not running in Colab
        return os.getenv("HF_TOKEN", None)

# Function to pre-download models
def download_all_models():
    print("Downloading all models...")
    _HF_TOKEN = get_hf_token()
    if not _HF_TOKEN:
        raise ValueError("HF_TOKEN is not available. Please set it in Colab secrets or environment variables.")

    # Download the pipelines
    for model_key, config in models.items():
        try:
            pipeline_class = config["pipeline_class"]
            model_id = config["model_id"]
            # Download the pipeline (weights will be cached)
            # Remove use_auth_token from here, we'll use token directly
            pipeline_class.from_pretrained(model_id, token=_HF_TOKEN, **config.get("config", {}))
            print(f"Model '{model_key}' downloaded successfully.")
        except Exception as e:
            print(f"Error downloading model '{model_key}': {e}")
    print("Model download process complete.")

# Call the function to download all the models
download_all_models()