from huggingface_hub import snapshot_download
import os
import shutil

# -------------------------
# Config
# -------------------------
repo_id = "rhasspy/piper-voices"
language = "en_US"
speaker = "ryan"
model_size = "medium"

subfolder = f"en/{language}/{speaker}/{model_size}"
local_dir = os.path.join(os.getcwd(), "piper_model")
os.makedirs(local_dir, exist_ok=True)

# -------------------------
# Download model
# -------------------------
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=f"{subfolder}/**",
    local_dir=local_dir,
    local_dir_use_symlinks=False
)

# -------------------------
# Move ONNX file to consistent path
# -------------------------
downloaded_onnx = os.path.join(local_dir, subfolder, f"{speaker}-{model_size}.onnx")
target_path = os.path.join(local_dir, f"{speaker}-{model_size}.onnx")

if not os.path.exists(downloaded_onnx):
    raise FileNotFoundError(f"Downloaded model not found at {downloaded_onnx}")

shutil.move(downloaded_onnx, target_path)

print(f"Model downloaded and saved to: {target_path}")
