from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

def compute_wcss(X, labels):
    """WCSS: sum of squared distances to cluster centroids"""
    wcss = 0.0
    for label in np.unique(labels):
        if label == -1:
            continue
        cluster_points = X[labels == label]
        centroid = cluster_points.mean(axis=0)
        wcss += np.sum((cluster_points - centroid) ** 2)
    return wcss

def compute_dunn_index(X, labels):
    """
    Dunn Index:
    min inter-cluster distance / max intra-cluster distance
    """
    unique_labels = [l for l in np.unique(labels) if l != -1]

    if len(unique_labels) < 2:
        return np.nan

    # Intra-cluster distances
    intra_dists = []
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 1:
            intra_dists.append(np.max(pdist(cluster_points)))
        else:
            intra_dists.append(0.0)

    max_intra = max(intra_dists)

    # Inter-cluster distances
    inter_dists = []
    for i, l1 in enumerate(unique_labels):
        for l2 in unique_labels[i + 1:]:
            d = np.min(
                cdist(X[labels == l1], X[labels == l2])
            )
            inter_dists.append(d)

    min_inter = min(inter_dists)

    return min_inter / max_intra if max_intra > 0 else np.nan

class ClassificationMetrics:
    def __init__(self, accuracy, precision, recall, f1_score, roc_auc):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.roc_auc = roc_auc

def show_classification_report(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    print(f"Accuracy  : {accuracy:.4f}  (maximize)")
    print(f"Precision : {precision:.4f}  (maximize)")
    print(f"Recall    : {recall:.4f}  (maximize)")
    print(f"F1 Score  : {f1:.4f}  (maximize)")
    print()
    print(classification_report(y_true, y_pred))

    fig, (ax_cm, ax_roc) = plt.subplots(1, 2, figsize=(12, 5))

    cm = confusion_matrix(y_true, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_title("Confusion Matrix")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax_roc.plot([0,1], [0,1], 'k--')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend()

    plt.tight_layout()
    plt.show()

    return ClassificationMetrics(accuracy, precision, recall, f1, roc_auc)


def show_clustering_report(X, labels):
    """
    Shows clustering report with a histogram of cluster sizes and a scatter plot of the clusters.

    :param X: a numpy array of shape (n_samples, n_features)
    :param labels: a numpy array of shape (n_samples,) containing cluster labels
    :return: None
    """

    unique_labels = sorted(set(labels))
    has_noise = -1 in unique_labels

    # ---------------------------
    # METRICS (exclude noise)
    # ---------------------------
    mask = labels != -1 if has_noise else np.ones(len(labels), dtype=bool)
    X_clean = X[mask]
    labels_clean = labels[mask]

    wcss = compute_wcss(X_clean, labels_clean)
    silhouette = silhouette_score(X_clean, labels_clean)
    db_index = davies_bouldin_score(X_clean, labels_clean)
    ch_index = calinski_harabasz_score(X_clean, labels_clean)
    dunn_index = compute_dunn_index(X_clean, labels_clean)
    print("Clustering Metrics")
    print(f"  WCSS (compactness)      : {wcss:.4f}  (minimize)")
    print(f"  Silhouette Score        : {silhouette:.4f}  (maximize)")
    print(f"  Davies-Bouldin Index    : {db_index:.4f}  (minimize)")
    print(f"  Calinski-Harabasz Index : {ch_index:.4f}  (maximize)")
    print(f"  Dunn Index              : {dunn_index:.4f}  (maximize)")

    # ---------------------------
    # LEFT PLOT: HISTOGRAM
    # ---------------------------
    counts = [np.sum(labels == label) for label in unique_labels]
    fig, (ax_hist, ax_scatter) = plt.subplots(1, 2, figsize=(14, 6))
    ax_hist.bar([str(l) for l in unique_labels], counts)
    ax_hist.set_title("Cluster Sizes")
    ax_hist.set_xlabel("Cluster Label")
    ax_hist.set_ylabel("Number of Points")
    ax_hist.grid(True, axis='y', linestyle='--', alpha=0.5)

    # ---------------------------
    # RIGHT PLOT: SCATTER
    # ---------------------------
    for label in unique_labels:
        mask = (labels == label)

        if label == -1:
            ax_scatter.scatter(
                X[mask, 0], X[mask, 1],
                s=25, c="black", marker="x", label="Noise"
            )
        else:
            ax_scatter.scatter(
                X[mask, 0], X[mask, 1],
                s=25, label=f"Cluster {label}", edgecolors='k', linewidth=0.3
            )

    ax_scatter.set_title("Clustering Results")
    ax_scatter.set_xlabel("X")
    ax_scatter.set_ylabel("Y")
    ax_scatter.grid(True)
    ax_scatter.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_feature_importances(model, feature_names, top_n=None):
    """
    Plot sorted feature importances of a fitted model.
    
    :param model: fitted RandomForestClassifier or RandomForestRegressor
    :param feature_names: list of feature names
    :param top_n: number of top features to show (all if None)
    :param figsize: size of the figure
    :param title: plot title
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)  # descending order

    if top_n is not None:
        indices = indices[:top_n]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], color='skyblue', edgecolor='black')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Importance")
    plt.title("Feature Importances")

    # Add numeric values to the bars
    for i, idx in enumerate(indices):
        plt.text(importances[idx] + 0.01*importances.max(), i, f"{importances[idx]:.3f}", 
                 va='center', fontsize=9)

    plt.tight_layout()
    plt.show()


def test_show_classification_report():
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1, 1, 0])
    show_classification_report(y_true, y_pred)

def test_show_clustering_report():
    X = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [5, 6],
        [8, 8],
        [8, 9],
        [25, 80],
        [24, 79],
        [23, 78],
        [22, 77]
    ])
    y_pred = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2, -1])  # -1 indicates noise
    show_clustering_report(X, y_pred)


def test_plot_feature_importances():
    data = load_iris()
    X = data.data
    y = data.target
    feature_names = data.feature_names

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)

    plot_feature_importances(model, feature_names, top_n=4)    

if __name__ == "__main__":
    test_plot_feature_importances()