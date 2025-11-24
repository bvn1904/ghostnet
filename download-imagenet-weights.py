import timm
import torch

# 'ghostnet_100' corresponds to the 1.0x width multiplier from the paper
ghostnet = timm.create_model('ghostnet_100', pretrained=True)

# Save the state dict to a file
torch.save(ghostnet.state_dict(), 'ghostnet_1x_imagenet.pth')
print("saved imagenet weights...")
