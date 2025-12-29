
# General
- create a utils file to be used across modules for :
    - dimentinality reduction : `pca_reduce`, 
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
        - add feature importance plot
    - `decision_tree`
        - add feature importance plot