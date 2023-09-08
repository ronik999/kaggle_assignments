from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import pickle
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
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


def perform_model_train(model, X_train, y_train, split_size, save_model, save_name):

    # add model, X_train, y_train and split size for cross-validation
    cv = StratifiedKFold(n_splits=split_size, random_state=1, shuffle=True)
    scoring = ['precision', 'recall', 'f1', 'roc_auc']
    cross_val = pd.DataFrame(cross_validate(model, X_train, y_train, scoring=scoring, cv=cv, n_jobs=-1, return_train_score=True))
    get_score_values(cross_val, model)
    if save_model == True:
        model.fit(X_train, y_train)
        filename = '../models/at1_week4_model/model_' + str(save_name) + ".sav"
        pickle.dump(model, open(filename, 'wb'))
        print("SAVED MODEL")

def plot_roc(X_val, y_val, model):
    no_score_probability = [0 for _ in range(len(y_val))]
    pred_probability = model.predict_proba(X_val)[:, 1]
    no_score_fpr, no_score_tpr, _ = roc_curve(y_val, no_score_probability)
    model_fpr, model_tpr, _ = roc_curve(y_val, pred_probability)
    # plot the roc curve for the model
    plt.plot(no_score_fpr, no_score_tpr, linestyle='--', label='No Skill')
    plt.plot(model_fpr, model_tpr, marker='.', label='')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
