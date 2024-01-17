import json
import glob
import os
from pathlib import Path
import random
import sys
import shutil 
from typing import List

import numpy as np
from tqdm import tqdm
from multiprocessing import Process, cpu_count

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lit_gpt.packed_dataset as packed_dataset
from lit_gpt import Tokenizer

import pandas as pd


def prepare_full(
    source_path: Path,
    checkpoint_dir: Path,
    destination_path: Path,
    chunk_size: int,
    filenames_subset: List[str] = None,
    process_id: int = 0
) -> None:
    import zstandard as zstd

    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(checkpoint_dir)

    # Use the provided filenames_subset or default to all filenames
    filenames = filenames_subset 
    
    if not filenames:
        raise RuntimeError(
            f"No files matching  found at {source_path}. \n"
            "Make sure you download the data..."
        )

    builder = packed_dataset.PackedDatasetBuilder(
        outdir=destination_path,
        prefix=f"train_pubmed_{process_id}",  # Use process_id to differentiate builders
        chunk_size=chunk_size,
        sep_token=tokenizer.bos_id,
        dtype="auto",
        vocab_size=tokenizer.vocab_size,
    )
    log_path = os.path.join(source_path, "log")
    wr = open(log_path, 'a')
    token_count = 0
    for filepath in filenames:
        with open(filepath, 'r') as f:
            for idx, line in enumerate(f):
                try:
                    json_line = json.loads(line)
                    text = json_line['text']
                except:
                    print(f"Error reading {filepath} idx={idx}")
                    continue
                text_ids = tokenizer.encode(text)
                token_count += len(text_ids)
                builder.add_array(np.array(text_ids, dtype=builder.dtype))
    wr.write(f"Total token count for process {process_id} = {token_count}\n")
    wr.close()
    # we throw away the final corpus to avoid meaningless corpus filled with bos_ids, see https://github.com/jzhang38/TinyLlama/issues/83 for more details
    # builder.write_reminder()

def write_to_jsonl(line, json_writer):
    json.dump(line, json_writer)
    json_writer.write("\n")

def prepare(
    source_path: Path = Path("../data/pubmed"),
    checkpoint_dir: Path = Path("checkpoints/microsoft/phi-2"),
    destination_path: Path = Path("../data/pubmed/processed"),
    chunk_size: int = 2049 * 1024,
    split_size: float=0.2,
    filenames_subset: List[str] = None,
) -> None:
    import time
    filenames = glob.glob(os.path.join(source_path, "*.jsonl"), recursive=True)
    # only retrain subsets that follow the prefix in filenames_subset
    if filenames_subset:
        filenames = [f for f in filenames if any([prefix in f for prefix in filenames_subset])]

    print("Processing files:")
    for fn in filenames:
        print(f"\t{fn}")

    if not os.path.exists(destination_path):
        os.makedirs(destination_path, exist_ok=True)
    else:
        print(f"Warning: Files already exist in {destination_path}.")

    num_processes = cpu_count()
    # evenly split data points by number of processes
    print(f"Evenly splitting dataset by processes...")
    start_time = time.time()
    json_writers = [open(os.path.join(destination_path, f"process{process}.jsonl"), "w") for process in range(num_processes)]
    idx = 0
    for fn in filenames:
        with open(fn, "r") as reader:
            for line_idx, line in enumerate(reader):
                try:
                    json_line = json.loads(line)
                    process_id = idx % num_processes
                    json_writer = json_writers[process_id]
                    write_to_jsonl(json_line, json_writer)
                    idx += 1
                except Exception as e:
                    print(e)
                    breakpoint()
                    print(f"Error reading {fn} idx={line_idx}")
                    continue

    for wr in json_writers:
        wr.close()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")



    # start chunking the dataset for each process
    print("Chunking dataset by blocks...")
    chunked_filenames = [os.path.join(destination_path, f"process{process}.jsonl") for process in range(num_processes)]
    chunked_filenames = np.array_split(chunked_filenames, num_processes)
    print(chunked_filenames)
    processes = []
    start_time = time.time()

    DEBUG = False
    if DEBUG:
        prepare_full(source_path, checkpoint_dir, destination_path, chunk_size, list(chunked_filenames[0]), 0)
        exit()

    for i, subset in enumerate(chunked_filenames):
        p = Process(target=prepare_full, args=(source_path, checkpoint_dir, destination_path, chunk_size, list(subset), i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

    # create file 
    os.makedirs(os.path.join(destination_path, "val"), exist_ok=True)
    os.makedirs(os.path.join(destination_path, "train"), exist_ok=True)
    train_val_files = glob.glob(os.path.join(destination_path, "*.bin"))

    # random split the train_val_files and move it to "val" directory
    random.shuffle(train_val_files)
    val_files = train_val_files[:int(len(train_val_files)*split_size)]
    for file in val_files:
        shutil.move(file, os.path.join(destination_path, "val"))
    
    train_files = train_val_files[int(len(train_val_files)*split_size):]
    for file in train_files:
        shutil.move(file, os.path.join(destination_path, "train"))

    val_files = glob.glob(os.path.join(destination_path, "val", "*.bin"))
    for file in val_files:
        new_file = file.replace("train", "val")
        os.rename(file, new_file)




if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(prepare)
