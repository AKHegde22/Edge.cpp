import os
from huggingface_hub import hf_hub_download

def download_model():
    model_id = "Qwen/Qwen1.5-0.5B-Chat-GGUF"
    filename = "qwen1_5-0_5b-chat-q4_k_m.gguf"
    
    print(f"Downloading {filename} from {model_id}...")
    local_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(local_dir, exist_ok=True)
    
    # Download the model
    model_path = hf_hub_download(
        repo_id=model_id,
        filename=filename,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    
    print(f"Model successfully downloaded to: {model_path}")

if __name__ == "__main__":
    download_model()
