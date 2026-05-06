from sklearn.metrics import accuracy_score, classification_report, f1_score

# same timeline as sampling (book_select.assign_era)
ERA_ORDER = [
    "Age of Reason",
    "Romantic",
    "Victorian",
    "Modernist",
    "Postmodern",
]


def evaluate_model(y_true, y_pred, label_col):
    print(f"\nLabel: {label_col}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")
    seen = set(y_true) | set(y_pred)
    labels = [e for e in ERA_ORDER if e in seen]
    for x in sorted(seen - set(labels)):
        labels.append(x)
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))


def get_metrics(y_true, y_pred, label_col):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
