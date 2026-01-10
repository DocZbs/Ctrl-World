#!/usr/bin/env python3
"""Download Ctrl-World model from HuggingFace."""

import os
from huggingface_hub import snapshot_download

# Disable proxy
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('HF_ENDPOINT', None)

# Download model
local_dir = "/data1/zbs_files/data/HF/hub/models--Youyi-Kou--ctrl-world"
print(f"Downloading Youyi-Kou/ctrl-world to {local_dir}...")

try:
    path = snapshot_download(
        repo_id="Youyi-Kou/ctrl-world",
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print(f"\n✓ Download complete!")
    print(f"Model saved to: {path}")
except Exception as e:
    print(f"\n✗ Download failed: {e}")
    import traceback
    traceback.print_exc()
