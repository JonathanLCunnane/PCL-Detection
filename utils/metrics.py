from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

def compute_metrics(labels, preds):
    """Compute F1, precision, recall."""
    f1 = f1_score(labels, preds, zero_division=0)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    return f1, precision, recall


def print_classification_report(labels, preds):
    print(classification_report(labels, preds, target_names=["Non-PCL", "PCL"]))
