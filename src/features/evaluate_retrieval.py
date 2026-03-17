"""Retrieval evaluation metrics for semi-synthetic anomaly detection."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate_retrieval(labels, scores, injection_log, top_k_list=None):
    if top_k_list is None:
        top_k_list = [10, 25, 50, 100]
    n_positive = int(labels.sum())
    ranked = np.argsort(-scores)
    precision_at_k = {}
    recall_at_k = {}
    for k in top_k_list:
        k_capped = min(k, len(labels))
        top_k_labels = labels[ranked[:k_capped]]
        tp = int(top_k_labels.sum())
        precision_at_k[k] = tp / k_capped if k_capped > 0 else 0.0
        recall_at_k[k] = tp / n_positive if n_positive > 0 else 0.0
    auroc = float(roc_auc_score(labels, scores)) if n_positive > 0 and n_positive < len(labels) else 0.0
    auprc = float(average_precision_score(labels, scores)) if n_positive > 0 else 0.0
    max_k = max(top_k_list)
    max_k_capped = min(max_k, len(labels))
    top_k_set = set(ranked[:max_k_capped].tolist())
    type_to_indices = {}
    for entry in injection_log:
        t = entry["type"]
        type_to_indices.setdefault(t, []).append(entry["index"])
    per_type_recall = {}
    for t, indices in type_to_indices.items():
        found = sum(1 for idx in indices if idx in top_k_set)
        per_type_recall[t] = found / len(indices) if indices else 0.0
    return {
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "auroc": auroc,
        "auprc": auprc,
        "per_type_recall_at_k": per_type_recall,
        "top_k_for_per_type": max_k,
    }
