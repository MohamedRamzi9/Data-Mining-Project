from sklearn.utils import resample
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

LABEL = "fire"

class SampledDataset:
    def __init__(self, method_names):
        self.method_names = method_names
        self.index = 0

        self.best_method_name = None
        self.best_X_train = None
        self.best_y_train = None
        self.best_X_test = None
        self.best_y_test = None
        self.best_score = 0.0

    def go_next_method(self, X_train, y_train, X_test, y_test, score):
        if self.index < len(self.method_names):
            if score > self.best_score:
                self.best_score = score
                self.best_X_train = X_train
                self.best_y_train = y_train
                self.best_X_test = X_test
                self.best_y_test = y_test
                self.best_method_name = self.method_names[self.index]
            self.index += 1
            return True
        return False

    def print_report(self, metric_name):
        print(f"Best sampling method: {self.best_method_name} with {metric_name}: {self.best_score:.4f}")
        print(f"New dataset sizes: Train={self.best_X_train.shape[0]}, Test={self.best_X_test.shape[0]}, Full={self.best_X_train.shape[0] + self.best_X_test.shape[0]}")
        print(f"Class distribution: class 0: {(self.best_y_train == 0).sum()}, class 1: {(self.best_y_train == 1).sum()}")


def random_under_sample(X, y, label_column=LABEL, random_state=42):
    """
    Perform random under-sampling to balance the dataset.

    :param X: pandas DataFrame or numpy array containing the features.
    :param y: pandas Series or numpy array containing the class labels.
    :param label_column: Name of the column containing class labels, by default "fire".
    :return: Balanced X DataFrame and y Series after under-sampling.
    """


    # Separate majority and minority classes
    majority_class = X[y == 0]
    minority_class = X[y == 1]

    # Downsample majority class
    majority_downsampled = resample(majority_class,
                                    replace=False,  # sample without replacement
                                    n_samples=len(minority_class),  # to match minority class
                                    random_state=42)  # reproducible results

    # Combine minority class with downsampled majority class
    balanced_X = pd.concat([majority_downsampled, minority_class])
    balanced_y = pd.Series([0] * len(majority_downsampled) + [1] * len(minority_class), name=label_column)

    return balanced_X, balanced_y

def tomek_under_sample(X, y, label_column=LABEL):
    """
    Perform Tomek Links under-sampling to balance the dataset.

    :param X: pandas DataFrame or numpy array containing the features.
    :param y: pandas Series or numpy array containing the class labels.
    :param label_column: Name of the column containing class labels, by default "fire".
    :return: Balanced X DataFrame and y Series after under-sampling.
    """

    # Apply Tomek Links
    tomek = TomekLinks()
    X_resampled, y_resampled = tomek.fit_resample(X, y)

    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled = pd.Series(y_resampled, name=label_column)

    return X_resampled, y_resampled

def random_over_sample(X, y, label_column=LABEL, random_state=42):
    """
    Perform random over-sampling to balance the dataset.

    :param X: pandas DataFrame or numpy array containing the features.
    :param y: pandas Series or numpy array containing the class labels.
    :param label_column: Name of the column containing class labels, by default "fire".
    :return: Balanced X DataFrame and y Series after over-sampling.
    """

    # Separate majority and minority classes
    majority_class = X[y == 0]
    minority_class = X[y == 1]
    n_extra_samples = len(majority_class) - len(minority_class)

    # Upsample minority class
    minority_upsampled = pd.concat([minority_class, resample(minority_class,
                                                            replace=True,  # sample with replacement
                                                            n_samples=n_extra_samples,  # to match majority class
                                                            random_state=42)])  # reproducible results

    # Combine majority class with upsampled minority class
    balanced_X = pd.concat([majority_class, minority_upsampled])
    balanced_y = pd.Series([0] * len(majority_class) + [1] * len(minority_upsampled), name=label_column)

    return balanced_X, balanced_y

def smote_over_sample(X, y, label_column=LABEL, random_state=42):
    """
    Perform SMOTE over-sampling to balance the dataset.

    :param X: pandas DataFrame or numpy array containing the features.
    :param y: pandas Series or numpy array containing the class labels.
    :return: Balanced X DataFrame and y Series after under-sampling.
    """

    # Apply SMOTE
    k_neighbors = min(5, sum(y == 1) - 1)  # Ensure k_neighbors is less than number of minority samples
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled = pd.Series(y_resampled, name=label_column)

    return X_resampled, y_resampled


def test_random_over_sample():
    df = pd.DataFrame({'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                       'fire': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]})
    balanced_df = random_over_sample(df.drop(columns=['fire']), df['fire'])
    print("Original class distribution:")
    print(df)
    print("\nBalanced class distribution after over-sampling:")
    print(balanced_df)
    
def test_random_under_sample():
    df = pd.DataFrame({'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                       'fire': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]})
    balanced_df = random_under_sample(df.drop(columns=['fire']), df['fire'])
    print("Original class distribution:")
    print(df)
    print("\nBalanced class distribution after under-sampling:")
    print(balanced_df)

def test_smote_over_sample():
    df = pd.DataFrame({'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                       'fire': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]})
    balanced_df = smote_over_sample(df.drop(columns=['fire']), df['fire'])
    print("Original class distribution:")
    print(df)
    print("\nBalanced class distribution after SMOTE over-sampling:")
    print(balanced_df)

def test_tomek_under_sample():
    df = pd.DataFrame({'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                       'fire': [0, 1, 0, 1, 0, 0, 0, 0, 1, 1]})
    balanced_df = tomek_under_sample(df.drop(columns=['fire']), df['fire'])
    print("Original class distribution:")
    print(df)
    print("\nBalanced class distribution after Tomek Links under-sampling:")
    print(balanced_df)

if __name__ == "__main__":
    test_tomek_under_sample()