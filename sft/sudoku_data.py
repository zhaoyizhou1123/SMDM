import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def preprocess_sudoku(tokenizer, max_prompt_length=256, max_response_length=256):
    train_dataset = []

    # data = []
    file_path = 'zzy1123/sudoku_sft'
    # with open(file_path, 'r') as f:
    #     for line in f:
    #         data.append(line)
    dataset = load_dataset(file_path, split="train")

    for data in dataset:
        # d = data[i]

        # if len(d.split('||')) != 2:
        #     continue
        # if len(d.split('||')[1].split('####')) != 2:
        #     continue

        # question, thought, answer = d.split('||')[0], d.split('||')[1].split('####')[0], d.split('####')[1]
        question = data['input']
        answer = data['output']

        question = tokenizer(question, return_tensors="pt")['input_ids'][0]
        if tokenizer.pad_token_id is not None:
            pad_token_id = tokenizer.pad_token_id
        else:
            pad_token_id = 0
        q_len = question.shape[-1]
        q_padding = torch.full((max_prompt_length - q_len,), pad_token_id, dtype=question.dtype)
        question = torch.cat((q_padding, question), dim=-1) # left padding question
        # thought = tokenizer(thought, return_tensors="pt")['input_ids'][0]
        answer = tokenizer(answer, return_tensors="pt")['input_ids'][0]
        # answer = torch.cat((answer, torch.tensor([tokenizer.eos_token_id])), dim=-1)

        ans_length = answer.shape[-1]
        if ans_length > max_response_length:
            # exclude prompts that are too long
            continue
        ans_padding = torch.full((max_response_length - ans_length,), pad_token_id, dtype=answer.dtype)
        answer = torch.cat((answer, ans_padding), dim=-1)

        padded_data = torch.cat((question, answer), dim=-1)
        train_dataset.append(dict(data=padded_data, input_length=max_prompt_length,
                                  length=max_prompt_length + ans_length))


    train_dataset = CustomDataset(train_dataset)
    print(f"Final Sudoku dataset size: {len(train_dataset)}")
    return train_dataset