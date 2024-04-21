# Deep-Features-PCa
Script for the evaluation of the integration of deep features into a radiomics pipeline for prostate cancer aggressiveness classification.

In order to run the script as is, the dataset should come in a tabular format with the following structure:

| ID   | features       | Target |
|------|----------------|--------|
| idx1 | f1,f2,f3 (...) | 1      |

This can be easily changed by altering the python file.

All results form the nested cross-validation ( inner grid search and outer error estimation ) + training + holdout-test set are logged in WANDB for easy access and storage.
By saving the TP, FP, TN, FN it makes it possible to compute whatever non-probability based metrics that are required.
