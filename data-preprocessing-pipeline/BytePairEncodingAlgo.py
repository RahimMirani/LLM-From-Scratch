import importlib
import tiktoken

#tiktoken library for tokens, used by initial gpt too

#Using gpt2 tokenizer model
tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello how are you? <|endoftext|> Do you like tea? <|endoftext|> In the sunlit Terraces"
    "of someunknownPLace."
)

#Encoder
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print(integers)

#Decorder
decoded_strings = tokenizer.decode(integers)

print(decoded_strings)