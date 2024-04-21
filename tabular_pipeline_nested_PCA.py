import pandas as pd
import numpy as np
import re
import os
import wandb
from pathlib import Path
from glob import glob
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, cross_validate
from sklearn.metrics import make_scorer, cohen_kappa_score, fbeta_score, roc_auc_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from skrebate import ReliefF
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import argparse

def confusion_matrix_scorer(clf, X, y):
    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)
    f2 = fbeta_score(y,y_pred,beta=2)
    kappa = cohen_kappa_score(y,y_pred)
    auc = roc_auc_score(y,y_proba[:,1])
    cm = confusion_matrix(y, y_pred)
    return {'tn': cm[0, 0], 'fp': cm[0, 1],
            'fn': cm[1, 0], 'tp': cm[1, 1],
            'f2':f2,'kappa':kappa,'auc':auc}

parser = argparse.ArgumentParser()

# I/O params
parser.add_argument(
    '--data',dest='data',type=str,
    help="Path to the dataset csv",
    required=True)
parser.add_argument(
    '--folds',dest='n_folds',type=int,
    help="Number of folds for cross-validation",
    required=True)
parser.add_argument(
    '--fs',dest='fs',type=str,
    help="Feature selection type [PCA, Relief, Hybrid, None]",
    required=True)
parser.add_argument(
    '--clf',dest='clf',type=str,
    help="classifier to run",
    required=True)


# Value params
parser.add_argument(
    '--seed',dest='seed',type=int,default=42,
    help="Random seed for determinism")
parser.add_argument(
    '--project',dest='project',type=str, default='latent',
    help="Wandb project name")
parser.add_argument(
    '--run_name',dest='run_name',default=False, action='latent-run',
    help="name of the wandb run")

args = parser.parse_args()

def main():
    
    # random state
    rs = args.seed
    
    # number of cv folds
    number_folds = args.n_folds
    
    # Get the data

    ds = pd.read_csv(args.data)

    clf = args.clf

    X, Y = ds.drop(columns=['Target', 'ID']).values, ds['Target'].values

    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.3, random_state=args.seed, stratify=Y)
    
    sc = StandardScaler()
    X = sc.fit_transform(X)
    X_test = sc.transform(X_test)

    fs_method = None

    if args.fs == 'PCA':

        pca = PCA(random_state=42).fit(X)

        components = np.arange(1, X.shape[0]+1, step=1)
        variances = np.cumsum(pca.explained_variance_ratio_)

        n_components = [(c,v) for (c,v) in zip(components,variances) if v > 0.95 and v < 0.99][0]
        
        pca = PCA(random_state=42, n_components = n_components[0])
        X = pca.fit_transform(X)
        X_test = pca.transform(X_test)

    elif args.fs == 'Relief':
        fs_method = SelectFromModel(ReliefF(n_features_to_select=10, n_neighbors=5, n_jobs=-1))

    elif args.fs == 'Hybrid':

        pca = PCA(random_state=42).fit(X)

        components = np.arange(1, X.shape[0]+1, step=1)
        variances = np.cumsum(pca.explained_variance_ratio_)

        n_components = [(c,v) for (c,v) in zip(components,variances) if v > 0.95 and v < 0.99][0]
        
        pca = PCA(random_state=42, n_components = n_components[0])
        X = pca.fit_transform(X)
        X_test = pca.transform(X_test)


        if n_components[0] > 10:
            fs_method = SelectFromModel(ReliefF(n_features_to_select=10, n_neighbors=5, n_jobs=-1))

    
    # get class weights
    class_weights = ds['Target'].value_counts().to_dict()

    #kappa_scorer = make_scorer(cohen_kappa_score)
    f2_scorer = make_scorer(fbeta_score, beta=2)
    #auc_scorer = make_scorer(roc_auc_score)
    #cf_scorer = make_scorer(confusion_matrix_scorer)

    
    scoring = confusion_matrix_scorer

    wandb.init(project=args.project, name=args.run_name, settings=wandb.Settings(_disable_stats=True))
    
    if clf == 'dt':
        
        
        # first, perform nested cv+gs
        if fs_method != None:
            pipe_dt = make_pipeline(StandardScaler(),
                        VarianceThreshold(),
                        fs_method,
                        DecisionTreeClassifier(random_state=rs))
        else:
            pipe_dt = make_pipeline(StandardScaler(),
                        VarianceThreshold(),
                        DecisionTreeClassifier(random_state=rs))


        params = {
            'decisiontreeclassifier__max_depth': [1,2,3,4,5],
            'decisiontreeclassifier__min_samples_leaf': [5, 10, 20, 50],
            'decisiontreeclassifier__criterion': ["gini", "entropy"]
        }

        gs_dt = GridSearchCV(estimator=pipe_dt, 
                            param_grid=params, 
                            scoring=f2_scorer, 
                            refit=True,
                            cv=number_folds,
                            n_jobs=10,
                            verbose=1,)

        scores_cv = cross_validate(gs_dt, X, Y, scoring=scoring, cv=number_folds, verbose=1, return_train_score=True)
        
        
        
        # second, perform gs and inference
        
        if fs_method != None:
            pipe_dt = make_pipeline(StandardScaler(),
                        VarianceThreshold(),
                        fs_method,
                        DecisionTreeClassifier(random_state=rs))
        else:
            pipe_dt = make_pipeline(StandardScaler(),
                        VarianceThreshold(),
                        DecisionTreeClassifier(random_state=rs))

        
        grid_search = GridSearchCV(estimator=pipe_dt, 
            param_grid=params, 
            scoring=scoring, 
            refit='kappa',
            cv=number_folds,
            n_jobs=10,
            verbose=1,
            return_train_score=True,)
        
        gs = grid_search.fit(X, Y)
        clf = gs

        bp = gs.best_params_
    

    if clf == 'lr':
        
        
        # first, perform nested cv+gs
        if fs_method != None:
            pipe_lr = make_pipeline(StandardScaler(),
                        VarianceThreshold(),
                        fs_method,
                        LogisticRegression(random_state=rs))
        else:
            pipe_lr = make_pipeline(StandardScaler(),
                    VarianceThreshold(),
                    LogisticRegression(random_state=rs))



        params = {
            'logisticregression__penalty': ['l1', 'l2'],
            'logisticregression__solver': ['liblinear'],
            'logisticregression__tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'logisticregression__C': [100, 10, 1.0, 0.1, 0.01],
            #'max_iter': [200],
        }

        gs_lr = GridSearchCV(estimator=pipe_lr, 
                            param_grid=params, 
                            scoring=f2_scorer, 
                            refit=True,
                            cv=number_folds,
                            n_jobs=10,
                            verbose=1,)

        scores_cv = cross_validate(gs_lr, X, Y, scoring=scoring, cv=number_folds, verbose=1, return_train_score=True)
        
        
        
        # second, perform gs and inference
        
        if fs_method != None:
            pipe_lr = make_pipeline(StandardScaler(),
                        VarianceThreshold(),
                        fs_method,
                        LogisticRegression(random_state=rs))
        else:
            pipe_lr = make_pipeline(StandardScaler(),
                    VarianceThreshold(),
                    LogisticRegression(random_state=rs))

        
        grid_search = GridSearchCV(estimator=pipe_lr, 
            param_grid=params, 
            scoring=scoring, 
            refit='kappa',
            cv=number_folds,
            n_jobs=10,
            verbose=1,
            return_train_score=True,)
        
        gs = grid_search.fit(X, Y)
        clf = gs

        bp = gs.best_params_



    if clf == 'rf':
        
        if fs_method != None: 
            pipe_rf = make_pipeline(StandardScaler(),
                        VarianceThreshold(),
                        fs_method,
                        RandomForestClassifier(random_state=rs))
        else:
            pipe_rf = make_pipeline(StandardScaler(),
                        VarianceThreshold(),
                        RandomForestClassifier(random_state=rs))


        params = {
            'randomforestclassifier__n_estimators': [200],
            'randomforestclassifier__max_depth': [1,2,3,5],
            'randomforestclassifier__min_samples_split': [2,6,10,20],
            'randomforestclassifier__min_samples_leaf': [3,5,10],
        }

        gs_rf = GridSearchCV(estimator=pipe_rf, 
                            param_grid=params, 
                            scoring=f2_scorer, 
                            refit=True,
                            cv=number_folds,
                            n_jobs=10,
                            verbose=1,)

        scores_cv = cross_validate(gs_rf, X, Y, scoring=scoring, cv=number_folds, verbose=1, return_train_score=True)

        # second, perform gs and inference

        if fs_method != None: 
            pipe_rf = make_pipeline(StandardScaler(),
                        VarianceThreshold(),
                        fs_method,
                        RandomForestClassifier(random_state=rs))
        else:
            pipe_rf = make_pipeline(StandardScaler(),
                        VarianceThreshold(),
                        RandomForestClassifier(random_state=rs))

        
        grid_search = GridSearchCV(estimator=pipe_rf, 
            param_grid=params, 
            scoring=scoring, 
            refit='kappa',
            cv=number_folds,
            n_jobs=10,
            verbose=1,
            return_train_score=True,)
        
        gs = grid_search.fit(X, Y)
        clf = gs

        bp = gs.best_params_


    if clf == 'ada':        
        
        # first, perform nested cv+gs
        if fs_method != None:
            pipe_ada = make_pipeline(StandardScaler(),
                        VarianceThreshold(),
                        fs_method,
                        AdaBoostClassifier(random_state=rs))
        else:
            pipe_ada = make_pipeline(StandardScaler(),
                        VarianceThreshold(),
                        AdaBoostClassifier(random_state=rs))


        params = {
            'adaboostclassifier__n_estimators': [200],
            'adaboostclassifier__learning_rate': [0.001, 0.01, 0.1, 1.0, 1.5],
            'adaboostclassifier__algorithm': ["SAMME", "SAMME.R"]
        }

        gs_ada = GridSearchCV(estimator=pipe_ada, 
                            param_grid=params, 
                            scoring=f2_scorer, 
                            refit=True,
                            cv=number_folds,
                            n_jobs=-1,
                            verbose=1,)

        scores_cv = cross_validate(gs_ada, X, Y, scoring=scoring, cv=number_folds, verbose=1, return_train_score=True)
        
        
        
        # second, perform gs and inference
        
        if fs_method != None:
            pipe_ada = make_pipeline(StandardScaler(),
                        VarianceThreshold(),
                        fs_method,
                        AdaBoostClassifier(random_state=rs))
        else:
            pipe_ada = make_pipeline(StandardScaler(),
                        VarianceThreshold(),
                        AdaBoostClassifier(random_state=rs))

        
        grid_search = GridSearchCV(estimator=pipe_ada, 
            param_grid=params, 
            scoring=scoring, 
            refit='kappa',
            cv=number_folds,
            n_jobs=10,
            verbose=1,
            return_train_score=True,)
        
        gs = grid_search.fit(X, Y)
        clf = gs

        bp = gs.best_params_
        
        

    wandb.log(bp)


    y_pred = clf.predict(X_test)
    y_pred_train = clf.predict(X)
    
    for i in range(number_folds):
        wandb.summary["inner_Te_k_f_"+str(i)] = scores_cv['test_kappa'][i]
        wandb.summary["inner_Tr_k_f_"+str(i)] = scores_cv['train_kappa'][i]
        wandb.summary["inner_Te_f2_f_"+str(i)] = scores_cv['test_f2'][i]
        wandb.summary["inner_Tr_f2_f_"+str(i)] = scores_cv['train_f2'][i]
        wandb.summary["inner_Te_auc_f_"+str(i)] = scores_cv['test_auc'][i]
        wandb.summary["inner_Tr_auc_f_"+str(i)] = scores_cv['train_auc'][i]
        
        wandb.summary["inner_Te_tn_f_"+str(i)] = scores_cv['test_tn'][i]
        wandb.summary["inner_Tr_tn_f_"+str(i)] = scores_cv['train_tn'][i]
        wandb.summary["inner_Te_fp_f_"+str(i)] = scores_cv['test_fp'][i]
        wandb.summary["inner_Tr_fp_f_"+str(i)] = scores_cv['train_fp'][i]
        wandb.summary["inner_Te_fn_f_"+str(i)] = scores_cv['test_fn'][i]
        wandb.summary["inner_Tr_fn_f_"+str(i)] = scores_cv['train_fn'][i]
        wandb.summary["inner_Te_tp_f_"+str(i)] = scores_cv['test_tp'][i]
        wandb.summary["inner_Tr_tp_f_"+str(i)] = scores_cv['train_tp'][i]
        
    for i in range(number_folds):
        wandb.summary["outer_Te_k_f_"+str(i)] = gs.cv_results_['split'+str(i)+'_test_kappa'][gs.best_index_]
        wandb.summary["outer_Tr_k_f_"+str(i)] = gs.cv_results_['split'+str(i)+'_train_kappa'][gs.best_index_]
        wandb.summary["outer_Te_f2_f_"+str(i)] = gs.cv_results_['split'+str(i)+'_test_f2'][gs.best_index_]
        wandb.summary["outer_Tr_f2_f_"+str(i)] = gs.cv_results_['split'+str(i)+'_train_f2'][gs.best_index_]
        wandb.summary["outer_Te_auc_f_"+str(i)] = gs.cv_results_['split'+str(i)+'_test_auc'][gs.best_index_]
        wandb.summary["outer_Tr_auc_f_"+str(i)] = gs.cv_results_['split'+str(i)+'_train_auc'][gs.best_index_]
        
        wandb.summary["outer_Te_tn_f_"+str(i)] = gs.cv_results_['split'+str(i)+'_test_tn'][gs.best_index_]
        wandb.summary["outer_Tr_tn_f_"+str(i)] = gs.cv_results_['split'+str(i)+'_train_tn'][gs.best_index_]
        wandb.summary["outer_Te_fp_f_"+str(i)] = gs.cv_results_['split'+str(i)+'_test_fp'][gs.best_index_]
        wandb.summary["outer_Tr_fp_f_"+str(i)] = gs.cv_results_['split'+str(i)+'_train_fp'][gs.best_index_]
        wandb.summary["outer_Te_fn_f_"+str(i)] = gs.cv_results_['split'+str(i)+'_test_fn'][gs.best_index_]
        wandb.summary["outer_Tr_fn_f_"+str(i)] = gs.cv_results_['split'+str(i)+'_train_fn'][gs.best_index_]
        wandb.summary["outer_Te_tp_f_"+str(i)] = gs.cv_results_['split'+str(i)+'_test_tp'][gs.best_index_]
        wandb.summary["outer_Tr_tp_f_"+str(i)] = gs.cv_results_['split'+str(i)+'_train_tp'][gs.best_index_]


    kappa = cohen_kappa_score(Y_test, y_pred)
    kappa_train = cohen_kappa_score(Y, y_pred_train)
    f2 = fbeta_score(Y_test, y_pred, beta=2)
    f2_train = fbeta_score(Y, y_pred_train, beta=2)
    auc = roc_auc_score(Y_test, y_pred)
    auc_train = roc_auc_score(Y, y_pred_train)
    tn_train, fp_train, fn_train, tp_train = confusion_matrix(Y, y_pred_train).ravel()
    tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()

    wandb.summary["kappa test"] = kappa
    wandb.summary["kappa train"] = kappa_train
    wandb.summary["f2 test"] = f2
    wandb.summary["f2 train"] = f2_train
    wandb.summary["auc test"] = auc
    wandb.summary["auc train"] = auc_train
    
    wandb.summary["tn test"] = tn
    wandb.summary["tn train"] = tn_train
    wandb.summary["fp test"] = fp
    wandb.summary["fp train"] = fp_train
    wandb.summary["fn test"] = fn
    wandb.summary["fn train"] = fn_train
    wandb.summary["tp test"] = tp
    wandb.summary["tp train"] = tp_train
    

    wandb.finish()
            
            
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
