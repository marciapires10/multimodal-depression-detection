import pandas as pd
import numpy as np

from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import f1_score, recall_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedKFold

df = pd.read_csv("../data_preprocessing/final_clean_transcripts.csv", index_col="Participant_ID")

# convert to occurrence matrix
cv = CountVectorizer(min_df=3)

X = cv.fit_transform(df.Transcript).toarray()
y = df.PHQ8_Binary


def train_test(X, y, test="../test_split_Depression_AVEC2017.csv"):
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


X_train, X_test, y_train, y_test = train_test(X, y)
X_train, y_train = shuffle(X_train, y_train, random_state=42)


def k_cross_validation(model, X=X_train, y=y_train):
    rkf = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)

    f1_score_res = []
    recall_res = []

    for train_index, test_index in rkf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        predict_values = model.predict(X_test)

        f1_scr = f1_score(y_test, predict_values)
        f1_score_res.append(f1_scr)

        recall = recall_score(y_test, predict_values)
        recall_res.append(recall)

    return f1_score_res, recall_res


def logistic_regression():

    model = LogisticRegression(n_jobs=3, C=10000)
    f1_score_res, recall_res = k_cross_validation(model)

    f1_val = np.mean(f1_score_res)
    recall_val = np.mean(recall_res)

    print(f"Logistic Regression\nBest F1 Score: {f1_val}\nBest Recall Score: {recall_val}\n")


logistic_regression()


def random_forest():

    model = RandomForestClassifier(random_state=42, n_estimators=1)
    f1_score_res, recall_res = k_cross_validation(model)

    f1_val = np.mean(f1_score_res)
    recall_val = np.mean(recall_res)

    print(f"Random Forest\nBest F1 Score: {f1_val}\nBest Recall Score: {recall_val}\n")


# random_forest()


def decision_tree():

    model = DecisionTreeClassifier(random_state=42, max_depth=4, min_samples_leaf=10)
    f1_score_res, recall_res = k_cross_validation(model)

    f1_val = np.mean(f1_score_res)
    recall_val = np.mean(recall_res)

    print(f"Decision Tree\nBest F1 Score: {f1_val}\nBest Recall Score: {recall_val}\n")

# decision_tree()


def svm():

    model = SVC(kernel='linear')
    f1_score_res, recall_res = k_cross_validation(model)

    f1_val = np.mean(f1_score_res)
    recall_val = np.mean(recall_res)

    print(f"SVM Linear\nBest F1 Score: {f1_val}\nBest Recall Score: {recall_val}\n")


# svm()

