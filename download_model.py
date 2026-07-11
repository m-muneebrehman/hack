import os
from huggingface_hub import hf_hub_download

def download_model():
    print("Downloading Qwen2.5-1.5B-Instruct Q4_K_M model...")
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = hf_hub_download(
        repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
        local_dir=model_dir,
    )
    print(f"Model downloaded to: {model_path}")

if __name__ == "__main__":
    download_model()
