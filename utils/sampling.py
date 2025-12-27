from sklearn.utils import resample
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

LABEL = "fire"


def random_under_sample(data, label_column=LABEL, random_state=42):
    """
    Perform random under-sampling to balance the dataset.

    :param data: pandas DataFrame containing the dataset.
    :param label_column: Name of the column containing class labels, by default "fire".
    :return: Balanced pandas DataFrame after under-sampling.
    """


    # Separate majority and minority classes
    majority_class = data[data[label_column] == 0]
    minority_class = data[data[label_column] == 1]

    # Downsample majority class
    majority_downsampled = resample(majority_class,
                                    replace=False,  # sample without replacement
                                    n_samples=len(minority_class),  # to match minority class
                                    random_state=42)  # reproducible results

    # Combine minority class with downsampled majority class
    balanced_data = pd.concat([majority_downsampled, minority_class])

    return balanced_data

def tomek_under_sample(data, label_column=LABEL):
    """
    Perform Tomek Links under-sampling to balance the dataset.

    :param data: pandas DataFrame containing the dataset.
    :param label_column: Name of the column containing class labels, by default "fire".
    :return: Balanced pandas DataFrame after Tomek Links under-sampling.
    """

    # Separate features and labels
    X = data.drop(columns=[label_column])
    y = data[label_column]

    # Apply Tomek Links
    tomek = TomekLinks()
    X_resampled, y_resampled = tomek.fit_resample(X, y)

    # Combine resampled features and labels into a DataFrame
    balanced_data = pd.DataFrame(X_resampled, columns=X.columns)
    balanced_data[label_column] = y_resampled

    return balanced_data

def random_over_sample(data, label_column=LABEL, random_state=42):
    """
    Perform random over-sampling to balance the dataset.

    :param data: pandas DataFrame containing the dataset.
    :param label_column: Name of the column containing class labels, by default "fire".
    :return: Balanced pandas DataFrame after over-sampling.
    """

    # Separate majority and minority classes
    majority_class = data[data[label_column] == 0]
    minority_class = data[data[label_column] == 1]
    n_extra_samples = len(majority_class) - len(minority_class)

    # Upsample minority class
    minority_upsampled = pd.concat([minority_class, resample(minority_class,
                                                            replace=True,  # sample with replacement
                                                            n_samples=n_extra_samples,  # to match majority class
                                                            random_state=42)])  # reproducible results

    # Combine majority class with upsampled minority class
    balanced_data = pd.concat([majority_class, minority_upsampled])

    return balanced_data

def smote_over_sample(data, label_column=LABEL, random_state=42):
    """
    Perform SMOTE over-sampling to balance the dataset.

    :param data: pandas DataFrame containing the dataset.
    :param label_column: Name of the column containing class labels, by default "fire".
    :return: Balanced pandas DataFrame after SMOTE over-sampling.
    """

    # Separate features and labels
    X = data.drop(columns=[label_column])
    y = data[label_column]

    # Apply SMOTE
    k_neighbors = min(5, sum(y == 1) - 1)  # Ensure k_neighbors is less than number of minority samples
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Combine resampled features and labels into a DataFrame
    balanced_data = pd.DataFrame(X_resampled, columns=X.columns)
    balanced_data[label_column] = y_resampled

    return balanced_data


def test_random_over_sample():
    df = pd.DataFrame({'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                       'fire': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]})
    balanced_df = random_over_sample(df)
    print("Original class distribution:")
    print(df)
    print("\nBalanced class distribution after over-sampling:")
    print(balanced_df)
    
def test_random_under_sample():
    df = pd.DataFrame({'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                       'fire': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]})
    balanced_df = random_under_sample(df)
    print("Original class distribution:")
    print(df)
    print("\nBalanced class distribution after under-sampling:")
    print(balanced_df)

def test_smote_over_sample():
    df = pd.DataFrame({'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                       'fire': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]})
    balanced_df = smote_over_sample(df)
    print("Original class distribution:")
    print(df)
    print("\nBalanced class distribution after SMOTE over-sampling:")
    print(balanced_df)

def test_tomek_under_sample():
    df = pd.DataFrame({'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                       'fire': [0, 1, 0, 1, 0, 0, 0, 0, 1, 1]})
    balanced_df = tomek_under_sample(df)
    print("Original class distribution:")
    print(df)
    print("\nBalanced class distribution after Tomek Links under-sampling:")
    print(balanced_df)

if __name__ == "__main__":
    test_tomek_under_sample()