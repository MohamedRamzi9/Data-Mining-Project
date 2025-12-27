
# General
- create a utils file to be used across modules for :
    - sampling functions : `random_under_sample`, `random_over_sample`, `smote_over_sample`, `tomek_under_sample` 
    - dimentinality reduction : `pca_reduce`, 
    - parameter tuning : `grid_search`, `bayesian_search`
    - metrics summary for supervised and unsupervised models
    - normalization functions : `normal_scale`, `standard_scale`, `robust_scale`
# Models
- `unsupervised`
    - add `clarans` model
    - `dbscan`
        - use reduction methods before clustering and compare results
        - add parameter tuning 
    - `kmeans`
        - use reduction methods before clustering and compare results
        - add parameter tuning 
    - use more metrics to evaluate clustering results
- `supervised`
    - `random_forest`
        - add parameter tuning
        - add sampling methods and compare results
        - add feature importance plot
    - `knn`
        - add parameter tuning
        - add sampling methods and compare results
    - `decision_tree`
        - add parameter tuning
        - add sampling methods and compare results
        - add feature importance plot