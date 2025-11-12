# Creating input-Target data pairs using Python data loaders.

import importlib
import tiktoken
from torch.utils.data import Dataset, DataLoader
import torch

#tiktoken library for tokens, used by initial gpt too

#Using gpt2 tokenizer model
tokenizer = tiktoken.get_encoding("gpt2")


with open ("the-verdict.txt", "r", encoding= "utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
#Total Vocabulary
print(len(enc_text))

#Setting the context to be 4 token
context_size = 4

x = enc_text[:context_size]
y = enc_text[1:context_size+1]

print (f"x:  {x}")
print (f"y:    {y}")

#Encoding
#setting up prediction pairs
for i in range(1, context_size+1):
    context = enc_text[:i]
    desired = enc_text[i]
    print (context, "--->", desired)


#Decoding the pairs
for i in range(1, context_size+1):
    context = enc_text[:i]
    desired = enc_text[i]

    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))


#Class to create tensors using Pytorch
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Using the sliding widow to chunk the book into overlapping sequences of max_lenght and generate tensors from it
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
