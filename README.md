# Building an LLM from Scratch
The dataset used in the whole implementation and repo is from the book "The Verdict." Its a toy dataset and it would scale the same way for larger datasets too. 

## LLM Tokenizer
1. Simple Tokenizer
2. Byte Pair encoding Algorithm

Steps in generating token embeddings:
Text --> broken down into sub words/characters --> converted into numbers through tokenizer (these numbers are random and does not capture semantic meaning) --> Passed into a Nueral Netwrok --> Token embeddings (these embeddings capture semetic meaning between words)