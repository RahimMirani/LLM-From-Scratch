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



# Creating the token embeddings and Positional embeddings for the orignal dataset
with open ("the-verdict.txt", "r", encoding= "utf-8") as f:
    raw_text = f.read()

vocab_size = 50257
output_dimension= 256
max_length = 4

token_embedding_layer = torch.nn.Embedding(vocab_size,output_dimension)
dataloader = create_dataloader_v1(
    raw_text, batch_size= 8, max_length= max_length, stride=max_length, shuffle=False
)

data_iter = iter(dataloader)
input, targets = next(data_iter)

#Converts this into 8x4x256 dimentional vector
token_embedding = token_embedding_layer(input)
print(token_embedding.shape)

#Creating positional embeddings
context_length = max_length
pos_embeddings_layer = torch.nn.Embedding(context_length, output_dimension)
pos_embeddings = pos_embeddings_layer(torch.arange(max_length))
print(pos_embeddings.shape)

# Adding position embeddings to token embeddings
input_embeddings = token_embedding + pos_embeddings
print(input_embeddings.shape) #The shape remains the same but the values are added up inside the tensors