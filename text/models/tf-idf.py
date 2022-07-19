import math
from collections import Counter

import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import f1_score, recall_score, make_scorer, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV

import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv("../data_preprocessing/final_clean_transcripts.csv", index_col="Participant_ID")


tfIdfVectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = tfIdfVectorizer.fit_transform(df.Transcript)
X_dense = X.todense()
y = df.PHQ8_Binary


def train_test(X, y, test="/home/marciapires/Desktop/multimodal-depression-detection/test_split_Depression_AVEC2017.csv"):
    test_participants = pd.read_csv(test)['participant_ID'].values
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    for i in range(y.shape[0]):
        participant_no = y.index[i]

        if participant_no in test_participants:
            X_test.append(X[i])
            y_test.append(y[participant_no])
        else:
            X_train.append(X[i])
            y_train.append(y[participant_no])

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


X_train, X_test, y_train, y_test = train_test(X_dense, y)
X_train = X_train.reshape((142, 79698))
X_test = X_test.reshape((45, 79698))

y_train = y_train.reshape(142)
y_test = y_test.reshape(45)

X_train, y_train = shuffle(X_train, y_train, random_state=42)


# oversampling

oversample = RandomOverSampler(sampling_strategy='minority')
X_train, y_train = oversample.fit_resample(X_train, y_train)
X_train, y_train = shuffle(X_train, y_train, random_state=42)
#print(Counter(y_train))


def k_cross_validation(model, grid, X=X_train, y=y_train):
    rkf = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)
    scoring = {"f1": make_scorer(f1_score), "recall": make_scorer(recall_score)}
    cv = GridSearchCV(model, grid, cv=rkf, scoring=scoring, refit='f1')

    f1_score_res = []
    recall_res = []
    mae_res = []
    rmse_res = []

    for train_index, test_index in rkf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        cv.fit(X_train, y_train)
        predict_values = cv.predict(X_test)

        f1_scr = f1_score(y_test, predict_values)
        f1_score_res.append(f1_scr)

        recall = recall_score(y_test, predict_values)
        recall_res.append(recall)

        mae = mean_absolute_error(y_test, predict_values)
        mae_res.append(mae)

        mse = mean_squared_error(y_test, predict_values)
        rmse = math.sqrt(mse)
        rmse_res.append(rmse)

    return f1_score_res, recall_res, mae_res, rmse_res, cv.best_params_


def logistic_regression():

    grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"]}
    #model = LogisticRegression(n_jobs=3, C=10000)
    model = LogisticRegression(solver='liblinear')
    f1_score_res, recall_res, mae_res, rmse_res, best_params = k_cross_validation(model, grid)

    f1_val = np.mean(f1_score_res)
    recall_val = np.mean(recall_res)
    mae_val = np.mean(mae_res)
    rmse_val = np.mean(rmse_res)

    print(f"Logistic Regression\nBest F1 Score: {f1_val}\nBest Recall Score: {recall_val}\n"
          f"MAE: {mae_val}\nRMSE: {rmse_val}\n")
    print("Tuned hpyerparameters :", best_params)


#logistic_regression()


def random_forest():

    grid = {
        'n_estimators': [1, 10],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [4, 5, 6, 7, 8],
        'criterion': ['gini', 'entropy']
    }
    model = RandomForestClassifier(random_state=42)
    f1_score_res, recall_res, mae_res, rmse_res, best_params = k_cross_validation(model, grid)

    f1_val = np.mean(f1_score_res)
    recall_val = np.mean(recall_res)
    mae_val = np.mean(mae_res)
    rmse_val = np.mean(rmse_res)

    print(f"Random Forest\nBest F1 Score: {f1_val}\nBest Recall Score: {recall_val}\n"
          f"MAE: {mae_val}\nRMSE: {rmse_val}\n")
    print("Tuned hpyerparameters :", best_params)


#random_forest()


def decision_tree():
    grid = {
        'criterion': ['gini', 'entropy'], 'max_depth': [2, 4, 6, 8, 10, 12],
    }
    model = DecisionTreeClassifier(random_state=42, min_samples_leaf=10)
    f1_score_res, recall_res, mae_res, rmse_res, best_params = k_cross_validation(model, grid)

    f1_val = np.mean(f1_score_res)
    recall_val = np.mean(recall_res)
    mae_val = np.mean(mae_res)
    rmse_val = np.mean(rmse_res)

    print(f"Decision Tree\nBest F1 Score: {f1_val}\nBest Recall Score: {recall_val}\n"
          f"MAE: {mae_val}\nRMSE: {rmse_val}\n")
    print("Tuned hyperparameters :", best_params)


#decision_tree()


def svm():

    grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100]},
                        {'kernel': ['poly'], 'degree': [3, 4, 5], 'C': [1, 10, 100]},
                        {'kernel': ['linear'], 'C': [1, 10, 100]}]
    model = SVC()
    f1_score_res, recall_res, mae_res, rmse_res, best_params = k_cross_validation(model, grid)

    f1_val = np.mean(f1_score_res)
    recall_val = np.mean(recall_res)
    mae_val = np.mean(mae_res)
    rmse_val = np.mean(rmse_res)

    print(f"SVM \nBest F1 Score: {f1_val}\nBest Recall Score: {recall_val}\n"
          f"MAE: {mae_val}\nRMSE: {rmse_val}\n")
    print("Tuned hyperparameters :", best_params)


svm()
