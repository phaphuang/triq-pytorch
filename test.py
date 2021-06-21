import torch

batch_size = 4
nb_classes = 5
target = torch.empty(batch_size, dtype=torch.long).random_(nb_classes)
print(target.shape)
print(target)