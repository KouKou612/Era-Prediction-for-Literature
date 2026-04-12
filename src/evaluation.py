import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score


def decade_to_int(decade_str: str) -> int:
    return int(str(decade_str).replace("s", ""))


def evaluate_classification(y_true, y_pred, label_col: str) -> None:
    print(f"\nLabel: {label_col}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")
    print(classification_report(y_true, y_pred, zero_division=0))


def evaluate_decade_distance(y_true, y_pred) -> None:
    y_true_int = np.array([decade_to_int(x) for x in y_true])
    y_pred_int = np.array([decade_to_int(x) for x in y_pred])

    diffs = np.abs(y_pred_int - y_true_int)

    print("Decade distance evaluation:")
    print(f"Average decade error: {diffs.mean():.2f} years")
    print(f"Median decade error: {np.median(diffs):.2f} years")
    print(f"Within 10 years: {(diffs <= 10).mean():.4f}")
    print(f"Within 20 years: {(diffs <= 20).mean():.4f}")
    print(f"Within 30 years: {(diffs <= 30).mean():.4f}")


def evaluate_model(y_true, y_pred, label_col: str) -> None:
    evaluate_classification(y_true, y_pred, label_col)

    if label_col == "decade":
        print()
        evaluate_decade_distance(y_true, y_pred)


def get_metrics(y_true, y_pred, label_col: str) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    if label_col == "decade":
        y_true_int = np.array([decade_to_int(x) for x in y_true])
        y_pred_int = np.array([decade_to_int(x) for x in y_pred])
        diffs = np.abs(y_pred_int - y_true_int)

        metrics["avg_decade_error"] = float(diffs.mean())
        metrics["median_decade_error"] = float(np.median(diffs))
        metrics["within_10_years"] = float((diffs <= 10).mean())
        metrics["within_20_years"] = float((diffs <= 20).mean())
        metrics["within_30_years"] = float((diffs <= 30).mean())

    return metrics