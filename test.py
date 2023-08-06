import torch
from models import SimCLR
from torchvision.models import resnet18
# Create a new instance of the 
base_encoder = eval('resnet18')
proj_dim = 128
load_path = './simclr_resnet18_epoch50.pt'
model = SimCLR(base_encoder, projection_dim=proj_dim, pred_epochs=10).cuda()

# Save the initial values of the model parameters
initial_state_dict = model.state_dict()

# Load the saved state dictionary
model.load_state_dict(torch.load(load_path).state_dict())

# Compare the values of the model parameters before and after loading the state dictionary
for name, param in model.named_parameters():
    if not torch.equal(param, initial_state_dict[name]):
        print(f"Parameter {name} has changed")

# this is a test
