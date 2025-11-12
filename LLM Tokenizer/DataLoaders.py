# Creating input-Target data pairs using Python data loaders.

import importlib
import tiktoken
from torch.utils.data import Dataset, DataLoader

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