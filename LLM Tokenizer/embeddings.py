# Creating token/vector embeddings from token ids

import torch

input_ids = torch.tensor([2,3,5,1])
vocab_size = 6
output_dimension = 3 #dimension of the output vector

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dimension)
print(embedding_layer.weight)
    
