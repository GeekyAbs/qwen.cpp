from huggingface_hub import snapshot_download

REPO = "Qwen/Qwen2.5-0.5B-Instruct"
OUT = "../weights/qwen" 

path = snapshot_download(
    REPO,
    local_dir=OUT,
    allow_patterns=[
        "*.safetensors", 
        "config.json", 
        "tokenizer.json", 
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt"
    ],
)
print("Downloaded to:", path)