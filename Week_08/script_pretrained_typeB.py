import torch
import torchvision.models as models

# ----------- Part 1: Save the weights -----------

model = models.resnet18(weights='DEFAULT')
weights_path = "resnet18_weights.pth"
torch.save(model.state_dict(), weights_path)
print(f"Weights saved to {weights_path}, upload these weights to your kaggle notebook (right sidebar)")

# ----------- Part 2: Load the weights into a new model (do this in kaggle) -----------

new_model = models.resnet18()
new_model.load_state_dict(torch.load(weights_path))
new_model.eval()
