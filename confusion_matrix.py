import numpy as np
import pandas as pd
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import classification_report

def confusionMatrix(actual,predictions,class_names,task,cmap):
    confmat = ConfusionMatrix(num_classes=len(class_names), task=task)
    confmat_tensor = confmat(preds=predictions,target=actual)
    fig, ax = plot_confusion_matrix(
    cmap=cmap,
    conf_mat=confmat_tensor.numpy(), # matplotlib likes working with numpy
    class_names=class_names, # turn the row and column labels into class names
    figsize=(15, 15))
    
    ClassificationReport = classification_report(actual,predictions,output_dict=True)
    df = pd.DataFrame.from_dict(ClassificationReport)
    metrics = df.transpose()
    display(metrics)