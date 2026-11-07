# Checkpoint test metrics
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
    roc_auc_score, average_precision_score, log_loss,
)

def evaluate_on_test(model, test_loader, device):
    model.eval()
    y_pred_list, y_true_list, y_probs_list = [], [], []
    with torch.no_grad():
        for X_batch, y_batch, b_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).cpu()
            probs = torch.softmax(outputs, dim=1).numpy()
            y_probs_list.append(probs)
            y_pred_list.append(np.argmax(probs, axis=1))
            y_true_list.append(y_batch.numpy())

    y_pred = np.concatenate(y_pred_list)
    y_true = np.concatenate(y_true_list)
    y_probs = np.concatenate(y_probs_list)

    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["f1_score"] = f1_score(y_true, y_pred, average="macro")
    metrics["precision"] = precision_score(y_true, y_pred, average="macro")
    metrics["recall"] = recall_score(y_true, y_pred, average="macro")
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    metrics["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)

    try:
        metrics["auc_roc"] = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
    except ValueError:
        metrics["auc_roc"] = np.nan

    try:
        metrics["pr_auc"] = average_precision_score(np.eye(10)[y_true], y_probs, average="macro")
    except ValueError:
        metrics["pr_auc"] = np.nan

    try:
        metrics["log-loss"] = log_loss(y_true, y_probs)
    except Exception:
        metrics["log-loss"] = np.nan

    return metrics, y_true, y_pred
