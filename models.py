import torch
import torch.nn as nn
from ae_utils_exp import Disentangler, cifar10_norm, cifar10_inorm


class SimCLR(nn.Module):
    def __init__(self, base_encoder, disentangler=None, projection_dim=128):
        super().__init__()
        self.enc = base_encoder(weights=None)  # load model from torchvision.models without pretrained weights.
        self.feature_dim = self.enc.fc.in_features
        print(f"feature dimension:{self.feature_dim}")

        # Customize for CIFAR10. Replace conv 7x7 with conv 3x3, and remove first max pooling.
        # See Section B.9 of SimCLR paper.
        self.enc.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.enc.maxpool = nn.Identity()
        self.enc.fc = nn.Identity()  # remove final fully connected layer.

        # Add MLP projection.
        self.projection_dim = projection_dim
        self.projector = nn.Sequential(nn.Linear(self.feature_dim, 2048),
                                       nn.ReLU(),
                                       nn.Linear(2048, projection_dim))
        # From nashAE framework use method to disentangle representations
        self.disentangler = disentangler

    # Combining with NashAE framework, on each iteration train an autoEncoder and feed into foward the disentangle representations
    def forward(self, x):
        feature = self.enc(x)
        disentanged_projection = self.get_disentangled_projection(self.disentangler, feature)
        
        return feature, disentanged_projection
        
    def get_disentangled_projection(self, disentangler, dataset):
        # train predictors for n_epochs
        return disentangler.predict_z(dataset)

    

    

    