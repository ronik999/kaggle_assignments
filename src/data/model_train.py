from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import pickle
import warnings
import pandas as pd
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_score_values(scores, model_name):
    
    print("-"*60)
    print("TRAIN SCORES for " + str(model_name))
    print("-"*60)
    print("Precision Score: " + str(scores['train_precision'].mean()) + " +- " + str(scores['train_precision'].std()))
    print("Recall Score: " + str(scores['train_recall'].mean()) + " +- " + str(scores['train_recall'].std()))
    print("F1 Score: " + str(scores['train_f1'].mean()) + " +- " + str(scores['train_f1'].std()))
    print("AUC score: " + str(scores['train_roc_auc'].mean()) + " +- " + str(scores['train_roc_auc'].std()))
    print("-"*60)
    
    print("TEST SCORES for " + str(model_name))
    print("-"*60)
    print("Precision Score: " + str(scores['test_precision'].mean()) + " +- " + str(scores['test_precision'].std()))
    print("Recall Score: " + str(scores['test_recall'].mean()) + " +- " + str(scores['test_recall'].std()))
    print("F1 Score: " + str(scores['test_f1'].mean()) + " +- " + str(scores['test_f1'].std()))
    print("AUC score: " + str(scores['test_roc_auc'].mean()) + " +- " + str(scores['test_roc_auc'].std()))


def perform_model_train(model, X_train, y_train, split_size, save_model):

    # add model, X_train, y_train and split size for cross-validation
    cv = StratifiedKFold(n_splits=split_size, random_state=1, shuffle=True)
    scoring = ['precision', 'recall', 'f1', 'roc_auc']
    cross_val = pd.DataFrame(cross_validate(model, X_train, y_train, scoring=scoring, cv=cv, n_jobs=-1, return_train_score=True))
    get_score_values(cross_val, model)
    if save_model == True:
        model.fit(X_train, y_train)
        filename = '../models/at1_week3_model/model_' + str(model)[:5] + ".sav"
        pickle.dump(model, open(filename, 'wb'))
        print("SAVED MODEL")

