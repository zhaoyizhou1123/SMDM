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

def preprocess_ptr_follow(tokenizer, max_prompt_length=256, max_response_length=16):
    train_dataset = []

    # data = []
    file_path = 'zzy1123/ptr_follow_reverse_order_sft'
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

        # replace newlines with spaces to save token length
        answer = answer.replace('\n', ' ')

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
        # Remove bos token of answer
        if answer[0] == tokenizer.bos_token_id:
            answer = answer[1:] # The first token is not needed for tinyllama tokenizer
        # answer = torch.cat((answer, torch.tensor([tokenizer.eos_token_id])), dim=-1)

        ans_length = answer.shape[-1]
        if ans_length > max_response_length: # keep at least one space for eos token
            # exclude prompts that are too long
            print(f"Warning: answer length {ans_length} exceeds max_response_length {max_response_length}, skipping.")
            continue
        ans_padding = torch.full((max_response_length - ans_length,), tokenizer.eos_token_id, dtype=answer.dtype)
        answer = torch.cat((answer, ans_padding), dim=-1)

        padded_data = torch.cat((question, answer), dim=-1)
        train_dataset.append(dict(data=padded_data, input_length=max_prompt_length,
                                  length=max_prompt_length + ans_length))


    train_dataset = CustomDataset(train_dataset)
    print(f"Final Sudoku dataset size: {len(train_dataset)}")
    return train_dataset

def preprocess_ptr_follow_ar(tokenizer, max_prompt_length=256, max_response_length=16, r2l=False, order='reverse'):
    '''
    support different order.
    Pad token id is 1.
    '''

    train_dataset = []

    # data = []
    file_path = f'zzy1123/ptr_follow_{order}_order_sft'
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

        # replace newlines with spaces to save token length
        answer = answer.replace('\n', ' ')

        question = tokenizer(question, return_tensors="pt")['input_ids'][0]
        if tokenizer.pad_token_id is not None:
            pad_token_id = tokenizer.pad_token_id
        else:
            pad_token_id = 1
        q_len = question.shape[-1]
        q_padding = torch.full((max_prompt_length - q_len,), pad_token_id, dtype=question.dtype)
        question = torch.cat((q_padding, question), dim=-1) # left padding question
        # thought = tokenizer(thought, return_tensors="pt")['input_ids'][0]
        answer = tokenizer(answer, return_tensors="pt")['input_ids'][0]
        # Remove bos token of answer
        if answer[0] == tokenizer.bos_token_id:
            answer = answer[1:] # The first token is not needed for tinyllama tokenizer
        # answer = torch.cat((answer, torch.tensor([tokenizer.eos_token_id])), dim=-1)

        ans_length = answer.shape[-1]
        if ans_length > max_response_length: # keep at least one space for eos token
            # exclude prompts that are too long
            print(f"Warning: answer length {ans_length} exceeds max_response_length {max_response_length}, skipping.")
            continue
        ans_padding = torch.full((max_response_length - ans_length,), tokenizer.eos_token_id, dtype=answer.dtype)
        answer = torch.cat((answer, ans_padding), dim=-1)
        # reverse answer for r2l
        if r2l:
            answer = torch.flip(answer, dims=[0])

        padded_data = torch.cat((question, answer), dim=-1)
        train_dataset.append(dict(data=padded_data, input_length=max_prompt_length,
                                  length=max_prompt_length + ans_length))


    train_dataset = CustomDataset(train_dataset)
    print(f"Final Sudoku dataset size: {len(train_dataset)}")
    return train_dataset


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
                                              padding_side="right", use_fast=True)  

    dataset = preprocess_ptr_follow_ar(tokenizer, r2l=False)
    # print(dataset[0])
    data = dataset[0]['data']
    decoded_text = tokenizer.decode(data)
    print(decoded_text)

    # answer = "G T U Z P A C V I N W S F O J M"
    # tokenized_answer = tokenizer(answer, return_tensors="pt")['input_ids'][0]
    # print(tokenized_answer)
    # print(len(tokenized_answer))