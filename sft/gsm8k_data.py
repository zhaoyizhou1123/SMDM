import os
import time
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data, input_lengths, lengths):
        self.data = data                    # (N, max_length) int64
        self.input_lengths = input_lengths  # (N,) int64
        self.lengths = lengths              # (N,) int64

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return dict(data=self.data[idx], input_length=self.input_lengths[idx],
                    length=self.lengths[idx])


def preprocess_gsm8k(tokenizer, max_length=2048):
    cache_path = 'data/gsm8k/train_tokenized.pt'
    if os.path.exists(cache_path):
        t0 = time.time()
        print(f"Loading tokenized dataset from {cache_path}")
        cache = torch.load(cache_path, weights_only=True)
        print(f"Cache loaded in {time.time() - t0:.2f}s ({cache['data'].shape[0]} samples)")
        return CustomDataset(cache['data'], cache['input_length'], cache['length'])

    all_data = []
    all_input_lengths = []
    all_lengths = []

    raw = []
    file_path = 'data/gsm8k/train.txt'
    with open(file_path, 'r') as f:
        for line in f:
            raw.append(line)

    for d in raw:
        if len(d.split('||')) != 2:
            continue
        if len(d.split('||')[1].split('####')) != 2:
            continue

        question, thought, answer = d.split('||')[0], d.split('||')[1].split('####')[0], d.split('####')[1]
        question = 'Question: ' + question
        thought = 'Answer: ' + thought
        answer = '####' + answer

        question = tokenizer(question, return_tensors="pt")['input_ids'][0]
        thought = tokenizer(thought, return_tensors="pt")['input_ids'][0]
        answer = tokenizer(answer, return_tensors="pt")['input_ids'][0]
        answer = torch.cat((answer, torch.tensor([tokenizer.eos_token_id])), dim=-1)

        length1 = question.shape[-1] + thought.shape[-1]
        length2 = length1 + answer.shape[-1]
        if length2 > max_length:
            continue

        padding_length = max_length - length1
        padding = torch.full((padding_length,), tokenizer.eos_token_id, dtype=question.dtype)
        padded_data = torch.cat((question, thought, padding), dim=-1)
        all_data.append(padded_data)
        all_input_lengths.append(question.shape[-1])
        all_lengths.append(length1)

        padding_length = max_length - length2
        padding = torch.full((padding_length,), tokenizer.eos_token_id, dtype=question.dtype)
        padded_data = torch.cat((question, thought, answer, padding), dim=-1)
        all_data.append(padded_data)
        all_input_lengths.append(length1)
        all_lengths.append(length2)

    cache = dict(
        data=torch.stack(all_data),
        input_length=torch.tensor(all_input_lengths, dtype=torch.int64),
        length=torch.tensor(all_lengths, dtype=torch.int64),
    )
    torch.save(cache, cache_path)
    print(f"Saved tokenized dataset to {cache_path} ({cache['data'].shape[0]} samples)")
    return CustomDataset(cache['data'], cache['input_length'], cache['length'])
