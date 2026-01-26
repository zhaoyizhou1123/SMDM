import torch
from safetensors.torch import save_file
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Convert .pth model to .safetensors format")
    parser.add_argument("--pth_path", type=str, required=True, help="Path to the input .pth model file")
    parser.add_argument("--safetensor_path", type=str, required=True, help="Path to save the output .safetensors file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # finish if output file already exists
    if os.path.exists(args.safetensor_path):
        print(f"Output file {args.safetensor_path} already exists. Skipping conversion.")
    else:
        # 1. Load your .pth model
        # map_location="cpu" is safer for conversion scripts to avoid running out of GPU memory
        checkpoint = torch.load(args.pth_path, map_location="cpu")

        # 2. Extract the actual weights (Handling the nesting issue)
        # The error happened because your weights were inside the "model" key.
        if "model" in checkpoint:
            print(f"-> Found 'model' key in checkpoint. Extracting weights...")
            checkpoint = checkpoint["model"]
        elif "state_dict" in checkpoint:
            print(f"-> Found 'state_dict' key in checkpoint. Extracting weights...")
            checkpoint = checkpoint["state_dict"]

        # 3. Filter out non-tensor data
        # .safetensors strictly requires a dict of {string: Tensor}. 
        # This loop removes any leftover metadata (like 'epoch': int) that would cause a crash.
        tensors_to_save = {}
        for k, v in checkpoint.items():
            if isinstance(v, torch.Tensor):
                tensors_to_save[k] = v
            else:
                print(f"-> Skipping non-tensor key: {k} ({type(v)})")

        # 4. Save
        os.makedirs(os.path.dirname(args.safetensor_path), exist_ok=True)
        
        if len(tensors_to_save) == 0:
            print("Error: No tensors found to save! Check the input file structure.")
        else:
            save_file(tensors_to_save, args.safetensor_path)
            print(f"Successfully converted {args.pth_path} to {args.safetensor_path}")