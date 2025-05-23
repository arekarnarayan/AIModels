"""
Download Microsoft Phi-3-mini model to a local directory
"""

import os
from huggingface_hub import snapshot_download
from pathlib import Path

def download_model():
    # Create a directory for the model
    model_dir = Path("models/TinyLlama")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading model to {model_dir.absolute()}...")
    
    # Download the model from Hugging Face
    # The instruct version is optimized for conversation
    try:
        snapshot_download(
            repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            local_dir=str(model_dir),
            # Optional: include specific patterns to download
            # ignore_patterns=["*.safetensors.index.json"],
        )
        print(f"Successfully downloaded model to {model_dir.absolute()}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        
    # Check if download was successful
    if any(model_dir.iterdir()):
        print("Model files present in directory:")
        for file in model_dir.iterdir():
            print(f" - {file.name} ({file.stat().st_size / (1024*1024):.2f} MB)")
    else:
        print("No model files found. Download may have failed.")
    
    return model_dir

if __name__ == "__main__":
    model_path = download_model()
    print(f"\nModel download complete. Path: {model_path}")
    print("\nNote: If you encountered any authentication errors, try logging in:")
    print("  $ huggingface-cli login")