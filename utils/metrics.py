from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def show_classification_report(y_true, y_pred):
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average='binary'))
    print("Recall   :", recall_score(y_true, y_pred, average='binary'))
    print("F1 Score :", f1_score(y_true, y_pred, average='binary'))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    print(classification_report(y_true, y_pred))

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()


def show_clustering_report():
    pass  # Placeholder for clustering metrics implementation

if __name__ == "__main__":
    # Example usage
    y_true = [0, 1, 0, 1, 0, 1, 0, 0, 1, 1]
    y_pred = [0, 1, 0, 0, 0, 1, 1, 0, 1, 1]
    show_classification_report(y_true, y_pred)