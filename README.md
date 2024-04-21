# Deep-Features-PCa
Script for the evaluation of the integration of deep features into a radiomics pipeline for prostate cancer aggressiveness classification.

In order to run the script as is, the dataset should come in a tabular format with the following structure:

| ID   | feature1 | feature2 | (...) | Target |
|------|----------|----------|-------|--------|
| idx1 | 0.2      | 1        | (...) | 1      |

This can be easily changed by altering the python file.

All results form the nested cross-validation ( inner grid search and outer error estimation ) + training + holdout-test set are logged in WANDB for easy access and storage.
By saving the TP, FP, TN, FN it makes it possible to compute whatever non-probability based metrics that are required.
