from huggingface_hub import snapshot_download
import os

repo_id = "rhasspy/piper-checkpoints"
subfolder = "en/en_US/ryan/medium"

local_dir = os.path.join(os.getcwd(), "piper_model")

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=f"{subfolder}/**",
    local_dir=local_dir,
    local_dir_use_symlinks=False
)

print(f"Model downloaded to: {local_dir}")
