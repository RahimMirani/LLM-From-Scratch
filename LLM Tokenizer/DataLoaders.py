# Creating input-Target data pairs using Python data loaders.

import importlib
import tiktoken

#tiktoken library for tokens, used by initial gpt too

#Using gpt2 tokenizer model
tokenizer = tiktoken.get_encoding("gpt2")


with open ("the-verdict.txt", "r", encoding= "utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
#Total Vocabulary
print(len(enc_text))