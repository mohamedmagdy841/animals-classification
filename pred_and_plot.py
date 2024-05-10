import torch
from torch import nn
import torchvision
from torchvision import transforms
import cv2

import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

def pred_and_plot_image(model: torch.nn.Module, 
                        image_path: str, 
                        class_names= None, 
                        transform=None,
                        device: torch.device = device):
    """Makes a prediction on a target image and plots the image with its prediction."""
    
    # 1. Load in image and convert the tensor values to float32
    image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    if transform:
        target_image = transform(image)
    
    # 2. Make sure the model is on the target device
    model.to(device)
    
    # 3. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)
    
        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))
        
    # 4. Convert logits -> prediction probabilities (using torch.sigmoid() for binary-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred,dim=1)

    # 5. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    

    plt.imshow((image.squeeze().permute(1, 2, 0)).type(torch.uint8))
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title,fontsize=10)
    plt.axis(False);