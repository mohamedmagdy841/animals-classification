import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy

device = "cuda" if torch.cuda.is_available() else "cpu"

#metric = MulticlassAccuracy(num_classes=5).to(device)

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device=device):
    train_loss,train_acc=0,0
    model.to(device)
    model.train()
    for X,y in data_loader:
        X, y = X.to(device), y.to(device)
        y_logits = model(X)
        #y_pred = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
        loss = loss_fn(y_logits,y)
        train_loss += loss.item()
        #train_acc += metric(y_pred,y)
        #train_acc += accuracy_fn(y,y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    return train_loss

def test_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               device: torch.device=device):
    test_loss,test_accuracy=0,0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X,y in data_loader:
            X, y = X.to(device), y.to(device)
            y_logits = model(X)
            #y_pred = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
            loss = loss_fn(y_logits,y)
            test_loss += loss.item()
            #test_accuracy += metric(y_pred,y)
            #test_acc += accuracy_fn(y,y_pred)
    test_loss /= len(data_loader)
    #test_acc /= len(data_loader)
    return test_loss

from tqdm.auto import tqdm
# 1. Take in various parameters required for training and test steps
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int = 5):
    
    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "test_loss": []
    }
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model=model,
                                           data_loader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss = test_step(model=model,
            data_loader=test_dataloader,
            loss_fn=loss_fn)
        
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1}  | "
            f"train_loss: {train_loss:.4f} | "
            f"test_loss: {test_loss:.4f} | "
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        #results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        #results["test_acc"].append(test_acc)

    # 6. Return the filled results at the end of the epochs
    return results