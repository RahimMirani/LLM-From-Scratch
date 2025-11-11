# Creating input-Target data pairs using Python data loaders.

import importlib
import tiktoken

#tiktoken library for tokens, used by initial gpt too

#Using gpt2 tokenizer model
tokenizer = tiktoken.get_encoding("gpt2")