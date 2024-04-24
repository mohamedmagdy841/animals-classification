import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

def test(model,data_loader: torch.utils.data.DataLoader):
    y_list = []
    y_pred = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
            for X,y in data_loader:
                X, y = X.to(device), y.to(device)
                y_logits = model(X)
                target_image_pred_probs = torch.softmax(y_logits,dim=1)
                target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
                y_list.append(y.item())
                y_pred.append(target_image_pred_label.item())
    actual = torch.FloatTensor(y_list)
    predictions = torch.FloatTensor(y_pred)
    return actual,predictions