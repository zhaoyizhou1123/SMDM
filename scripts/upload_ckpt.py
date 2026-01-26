# Upload ckpt to huggingface hub
from huggingface_hub import HfApi
import os
import sys

if __name__ == "__main__":
    repo = 'zzy1123/smdm_ckpt'
    folder_path = "workdir/finetune"
    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo,
        repo_type="model",
        commit_message="upload smdm ckpt",
    )
