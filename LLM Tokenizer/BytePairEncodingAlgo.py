import importlib
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello how are you? <|endoftext|> Do you like tea? <|endoftext|> In the sunlit Terraces"
    "of someunknownPLace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print(integers)

decoded_strings = tokenizer.decode(integers)

print(decoded_strings)