import torch
from torchvision import datasets, transforms

# Load the CIFAR-10 training data
cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())


# Concatenate all the training data into a single tensor
data = torch.cat([d[0] for d in cifar_trainset])
print(f"cifar_trainset shape: {data.shape}")

# Calculate the mean and standard deviation along the channel dimension
mean = data.mean()
std = data.std()

print('Mean:', mean)
print('Standard deviation:', std)

# mean: 0.4734
# std: 0.2516