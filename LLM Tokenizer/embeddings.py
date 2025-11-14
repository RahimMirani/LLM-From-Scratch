# 1. Creating token/vector embeddings from token ids 
# 2. Creating Positional embeddings to capture word postion

import torch
from dataLoaders import create_dataloader_v1

# Creating Token Embeddings

input_ids = torch.tensor([2,3,5,1])

#Random 6 token ids
vocab_size = 6
output_dimension = 3 #dimension of the output vector

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dimension)
print(embedding_layer.weight)

# Looking up particular token ids from the embedding layer, this acts as a lookup of the the ids from the embeddings layer and returns weights/vectors of that ids
print(embedding_layer(input_ids))



# Creating Positional embeddings
vocab_size = 50257
output_dimension= 256
max_length = 4

token_embedding_layer = torch.nn.Embedding(vocab_size,output_dimension)
dataloader = create_dataloader_v1()



