import json
import glob
import os
from pathlib import Path
import sys
from typing import List
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, cpu_count

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lit_gpt.packed_dataset as packed_dataset
from lit_gpt import Tokenizer

# Filename for SlimPajama
slimpajama_sets = {
    "train": "train/chunk*/*",
    "validation": "validation/chunk*/*",
    "test": "test/chunk*/*",
}


def prepare_full(
    source_path: Path,
    tokenizer_path: Path,
    destination_path: Path,
    chunk_size: int,
    split: str = "train",
    filenames_subset: List[str] = None,
    process_id: int = 0
) -> None:
    # Only import zstandard if we actually need it to avoid errors if not installed
    # when processing plain jsonl files
    try:
        import zstandard as zstd
    except ImportError:
        zstd = None

    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(tokenizer_path)

    filenames = filenames_subset
    
    if not filenames:
        raise RuntimeError(
            f"No files provided to process."
        )

    builder = packed_dataset.PackedDatasetBuilder(
        outdir=destination_path,
        prefix=f"{split}_c4_{process_id}",
        chunk_size=chunk_size,
        sep_token=tokenizer.bos_id,
        dtype="auto",
        vocab_size=tokenizer.vocab_size,
    )

    for filepath in filenames:
        print(f"Processing {filepath}")
        
        # --- MODIFICATION START: Handle .jsonl vs .zst ---
        filepath_str = str(filepath)
        if filepath_str.endswith(".zst"):
            if zstd is None:
                raise ImportError("pip install zstandard is required to process .zst files")
            # Open compressed file
            file_context = zstd.open(open(filepath, "rb"), "rt", encoding="utf-8")
        else:
            # Open plain text file (standard jsonl)
            file_context = open(filepath, "rt", encoding="utf-8")
        # --- MODIFICATION END ---

        with file_context as f:
            for row in tqdm(f):
                try:
                    data = json.loads(row)
                    text = data["text"]
                    
                    # --- MODIFICATION: Safer metadata access ---
                    # This allows processing generic jsonl files that might not have 'meta'
                    meta = data.get("meta", {})
                    if meta.get("redpajama_set_name") == "RedPajamaGithub":
                        continue 
                    
                    text_ids = tokenizer.encode(text)
                    builder.add_array(np.array(text_ids, dtype=builder.dtype))
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line in {filepath}")
                    continue

    # builder.write_reminder()


def prepare(
    source_path: Path = Path("data/RedPajama-Data-1T-Sample"),
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    destination_path: Path = Path("data/red_pajama_sample"),
    chunk_size: int = 2049 * 1024,
    split: str = "train",
    percentage: float = 1.0,
    input_file: Path = None, # --- MODIFICATION: Added input_file argument ---
) -> None:
    import time

    # --- MODIFICATION START: Logic to choose between single file or directory scan ---
    if input_file:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        filenames = [str(input_file)]
        print(f"Processing single file: {input_file}")
    else:
        filenames = glob.glob(os.path.join(source_path, slimpajama_sets[split]), recursive=True)
        filenames = filenames[:int(len(filenames) * percentage)]
        print(f"Found {len(filenames)} files in {source_path}")
    # --- MODIFICATION END ---

    if not filenames:
        print("No files found to process.")
        return

    # If processing a single file, we only use 1 process to avoid empty processes
    num_processes = min(cpu_count(), len(filenames))
    chunked_filenames = np.array_split(filenames, num_processes)

    processes = []
    start_time = time.time()

    for i, subset in enumerate(chunked_filenames):
        # We pass list(subset) because numpy array slicing can be tricky with multiprocessing
        if len(subset) > 0:
            p = Process(target=prepare_full, args=(source_path, tokenizer_path, destination_path, chunk_size, split, list(subset), i))
            processes.append(p)
            p.start()

    for p in processes:
        p.join()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(prepare)