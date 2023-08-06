import torch
import torch.nn as nn
from ae_utils_exp import Disentangler, cifar10_norm, cifar10_inorm


class SimCLR(nn.Module):
    def __init__(self, base_encoder, projection_dim=128, pred_epochs = 0, batch_size = 100, lr = 0.01):
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
        self.disentangler = self.get_disentangler()
        self.pred_epochs = pred_epochs
        self.batch_size = batch_size
        self.lr = lr
    # Combining with NashAE framework, on each iteration train an autoEncoder and feed into foward the disentangle representations
    def forward(self, x):
        feature = self.enc(x)
        disentanged_projection = self.get_disentangled_projection(self.disentangler, feature)
        
        self.disentangler = self.get_disentangler() # revert disentangler to default params
        return feature, disentanged_projection
        
    def get_disentangled_projection(self, disentangler, dataset):
        # train predictors for n_epochs
        disentangler.fit(dataset=dataset, n_group=self.pred_epochs, batch_size=self.batch_size, pred_lr=self.lr)
        return disentangler.predict_z(dataset)
    
    def get_disentangler(self):
        n_lat = 128 # bottleneck 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        output = Disentangler(device=device, z_dim = n_lat)
        return output

    

    

    