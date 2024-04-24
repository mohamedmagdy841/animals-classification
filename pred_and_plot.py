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
    #target_image = io.imread(image_path)
    #target_image = cv2.imread(image_path)
    #rgb_image = cv2.cvtColor(target_image,cv2.COLOR_BGR2RGB)
    #plt.imshow(rgb_image.squeeze())
    #target_image = target_image.astype('float32') 

    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    #target_image = image / 255. 
    
    # 3. Transform if necessary
    if transform:
        target_image = transform(image)
    
    # 4. Make sure the model is on the target device
    model.to(device)
    
    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)
    
        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))
        
    # 6. Convert logits -> prediction probabilities (using torch.sigmoid() for binary-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred,dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    
    # get key from value
    #position = value_list.index(target_image_pred_label)
    #print(key_list[position])
    
    # 8. Plot the image alongside the prediction and prediction probability
    plt.imshow((image.squeeze().permute(1, 2, 0)).type(torch.uint8))
    #plt.imshow(target_image.squeeze().permute(1, 2, 0)) # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Class: {target_image_pred_label.max()} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else: 
        title = f"Pred: {key_list[position]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False);