# Building an LLM from Scratch
The dataset used in the whole implementation and repo is from the book "The Verdict." Its a toy dataset and it would scale the same way for larger datasets too. 

## Data Preprocessing Pipeline for LLM
Input Text (Data) --> Tokenization (Word based, Character based, Subword) --> Token Embeddings --> Positional Embeddings --> Input Embeddings(Token Embeddings + Positional Embeddings)

Steps in generating token embeddings:
Text --> broken down into sub words/characters --> converted into numbers (Token IDs) through tokenizer (these numbers are random and does not capture semantic meaning) --> Passed into a Nueral Netwrok --> Token embeddings vectors (these embeddings capture semetic meaning between words) --> Positional embeddings are generated and added to the token embeddings to capture the position of where the word appear, makes the llm more context aware ---> Data/Tokens fed into the LLM

File descriptions inside `data-preprocessing-pipeline`
1. `SimpleTokenizer.py` — A small, regex-based tokenizer and toy vocabulary builder. Shows how the dataset is split into tokens (words and punctuation), builds a `vocab` mapping, and provides `SimpleTokenizerV1` and `SimpleTokenizerV2` classes with `encode`/`decode` methods (the V2 class handles unknown tokens via `<|unk|>`). Includes examples of encoding/decoding and adding special tokens like `<|endoftext|>`.

2. `BytePairEncodingAlgo.py` — A short example demonstrating the use of the `tiktoken` (GPT-2) tokenizer. It encodes a sample text (including the `<|endoftext|>` special token) into token IDs and decodes them back to strings — useful as a quick reference for using `tiktoken` in this project.

3. `dataLoaders.py` — Reads `the-verdict.txt`, tokenizes the text with `tiktoken`, and constructs input-target pairs suitable for next-token prediction. Provides `GPTDatasetV1`, a PyTorch `Dataset` that chunks the token stream with a sliding window, and `create_dataloader_v1`, a helper that returns a `DataLoader`. The file also prints example batches to demonstrate the batching behavior.

4. `embeddings.py` — Demonstrates creating token embedding and positional embedding layers in PyTorch, using a small `vocab_size` and `output_dimension` for illustration. It shows how to look up token embeddings, build positional embeddings, and combine them (token + positional) to form input embeddings. This file uses `create_dataloader_v1` to obtain a batch and convert token IDs into dense vectors.


## Attention Mechanism
Input Embeddings(Final Embeddings generated above) --> Passed to `Attention Mechanism` --> Contextual Embeddings

1. Simplified Attention Mechanism
2. Self Attention Mechanism
3. Casual Self Attention Mechanism
4. Multi Head Attention Mechanism