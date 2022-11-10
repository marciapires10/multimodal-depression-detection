import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

df = pd.read_csv("../data_preprocessing/final_clean_transcripts5.csv", index_col="Participant_ID")
# print(df.info())


# convert to occurrence matrix
cv = CountVectorizer(min_df=3, stop_words="english")

# X = cv.fit_transform(df.Transcript).toarray()
X = df.Transcript
y = df.PHQ8_Binary


# set seed for reproducibility
# def set_seed(seed):
#     np.random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     tf.random.set_seed(seed)
#
#
# set_seed(42)


def train_test(X, y,
               test="/home/marciapires/Desktop/multimodal-depression-detection/test_split_Depression_AVEC2017.csv"):
    test_participants = pd.read_csv(test)['participant_ID'].values
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    for i in range(y.shape[0]):
        participant_no = y.index[i]

        if participant_no in test_participants:
            X_test.append(X[participant_no])
            y_test.append(y[participant_no])
        else:
            X_train.append(X[participant_no])
            y_train.append(y[participant_no])

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


X_train, X_test, y_train, y_test = train_test(X, y)
X_train, y_train = shuffle(X_train, y_train, random_state=42)

X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)


# Select features with highest f_classif statistics
classif_selector = SelectKBest(score_func=f_classif, k=500)
X_train = classif_selector.fit_transform(X_train, y_train)
X_test = classif_selector.transform(X_test)


# resampling

def apply_resampling(option, X_train, y_train):
    if option == "oversampling":
        resample = RandomOverSampler(sampling_strategy='minority')
    elif option == "undersampling":
        resample = RandomUnderSampler(sampling_strategy='majority')
    else:
        resample = SMOTE(random_state=42)
    X_train, y_train = resample.fit_resample(X_train, y_train)
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    return X_train, y_train


#X_train, y_train = apply_resampling("oversampling", X_train, y_train)


# # save data sets
# train_x = open('X_train_bow.pickle', 'wb')
# pickle.dump(X_train, train_x)
# train_x.close()
#
# train_y = open('y_train_bow.pickle', 'wb')
# pickle.dump(y_train, train_y)
# train_y.close()
#
# test_x = open('X_test_bow.pickle', 'wb')
# pickle.dump(X_test, test_x)
# test_x.close()
#
# test_y = open('y_test_bow.pickle', 'wb')
# pickle.dump(y_test, test_y)
# test_y.close()


def k_cross_validation(model, grid, X=X_train, y=y_train):
    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

    cv = GridSearchCV(model, grid, cv=rkf, refit=True, n_jobs=-1)

    f1_score_res = []
    recall_res = []
    precision_res = []

    for train_index, val_index in rkf.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        X_train, y_train = apply_resampling("smote", X_train, y_train)

        cv.fit(X_train, y_train)
        predict_values = cv.predict(X_val)
        f1_scr = f1_score(y_val, predict_values)
        f1_score_res.append(f1_scr)

        recall = recall_score(y_val, predict_values)
        recall_res.append(recall)

        precision = precision_score(y_val, predict_values)
        precision_res.append(precision)

    f1_val = np.mean(f1_score_res)
    recall_val = np.mean(recall_res)
    precision_val = np.mean(precision_res)

    print(f"Training...\nBest F1 Score: {f1_val}\nBest Recall Score: {recall_val}\nBest Precision Score: {precision_val}\n")
    print("Tuned hyperparameters: ", cv.best_params_)
    print("Best estimator: ", cv.best_estimator_)

    grid_predictions = cv.predict(X_test)
    print("predictions: ", grid_predictions)

    f1_scr = f1_score(y_test, grid_predictions)
    recall = recall_score(y_test, grid_predictions)
    precision = precision_score(y_test, grid_predictions)

    print("F1-Score: ", f1_scr)
    print("Recall: ", recall)
    print("Precision: ", precision)

    # save best model
    # file = open(str(model) + '_bow', 'wb')
    # pickle.dump(cv.best_estimator_, file)
    # file.close()

    predictions = evaluate(cv.best_estimator_)

    return predictions


def evaluate(best_model):

    grid_predictions = best_model.predict(X_test)
    print("predictions: ", grid_predictions)


    f1_scr = f1_score(y_test, grid_predictions)
    recall = recall_score(y_test, grid_predictions)
    precision = precision_score(y_test, grid_predictions)

    print("F1-Score: ", f1_scr)
    print("Recall: ", recall)
    print("Precision: ", precision)

    return grid_predictions


def logistic_regression():
    grid = [{'C': np.logspace(-3, 3, 7), 'penalty': ['l1'], 'solver': ['liblinear', 'sag', 'saga']},
           {'C': np.logspace(-3, 3, 7), 'penalty': ['l2'], 'solver': ['lbfgs', 'newton-cg']}]

    model = LogisticRegression(random_state=42)

    predictions = k_cross_validation(model, grid)

    # try:
    #     file = open(str(model) + '_bow', 'rb')
    #     final_model = pickle.load(file)
    #     file.close()
    #
    #     predictions = evaluate(final_model)
    #
    # except (OSError, IOError) as e:
    #     k_cross_validation(model, grid)
    #
    #     file = open(str(model) + '_bow', 'rb')
    #     final_model = pickle.load(file)
    #     file.close()
    #
    #     predictions = evaluate(final_model)

    return predictions


# lr_predictions = logistic_regression()
# print(lr_predictions)


def random_forest():
    grid = {
        'n_estimators': [1, 10, 100, 200],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 6, 7, 8, 9, 10],
        #'criterion': ['gini', 'entropy']
    }
    model = RandomForestClassifier(random_state=42)

    predictions = k_cross_validation(model, grid)

    # try:
    #     file = open(str(model) + '_bow', 'rb')
    #     final_model = pickle.load(file)
    #     file.close()
    #
    #     predictions = evaluate(final_model)
    #
    # except (OSError, IOError) as e:
    #     k_cross_validation(model, grid)
    #
    #     file = open(str(model) + '_bow', 'rb')
    #     final_model = pickle.load(file)
    #     file.close()
    #
    #     predictions = evaluate(final_model)

    return predictions


# rf_predictions = random_forest()
# print(rf_predictions)


def decision_tree():
    grid = {
        'criterion': ['gini', 'entropy'], 'max_depth': [2, 4, 6, 8, 10, 12, 14, 16], 'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
    model = DecisionTreeClassifier(random_state=42)

    predictions = k_cross_validation(model, grid)

    # try:
    #     file = open(str(model) + '_bow', 'rb')
    #     final_model = pickle.load(file)
    #     file.close()
    #
    #     predictions = evaluate(final_model)
    #
    # except (OSError, IOError) as e:
    #     k_cross_validation(model, grid)
    #
    #     file = open(str(model) + '_bow', 'rb')
    #     final_model = pickle.load(file)
    #     file.close()
    #
    #     predictions = evaluate(final_model)

    return predictions

# dt_predictions = decision_tree()
# print(dt_predictions)


def svm():
    grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100]},
            {'kernel': ['poly'], 'C': [1, 10, 100]},
            {'kernel': ['linear'], 'C': [1, 10, 100]}]
    model = SVC(random_state=42)

    predictions = k_cross_validation(model, grid)

    # try:
    #     file = open(str(model) + '_bow', 'rb')
    #     final_model = pickle.load(file)
    #     file.close()
    #
    #     predictions = evaluate(final_model)
    #
    # except (OSError, IOError) as e:
    #     k_cross_validation(model, grid)
    #
    #     file = open(str(model) + '_bow', 'rb')
    #     final_model = pickle.load(file)
    #     file.close()
    #
    #     predictions = evaluate(final_model)

    return predictions


# svm_predictions = svm()
# print(svm_predictions)


def knn():
    # knn_range = list(range(1, 31))
    grid = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'metric': ['euclidean', 'manhattan'],
            'weights': ['uniform', 'distance']}
    model = KNeighborsClassifier()


    predictions = k_cross_validation(model, grid)

    # try:
    #     file = open(str(model) + '_bow', 'rb')
    #     final_model = pickle.load(file)
    #     file.close()
    #
    #     predictions = evaluate(final_model)
    #
    # except (OSError, IOError) as e:
    #     k_cross_validation(model, grid)
    #
    #     file = open(str(model) + '_bow', 'rb')
    #     final_model = pickle.load(file)
    #     file.close()
    #
    #     predictions = evaluate(final_model)

    return predictions


# knn_predictions = knn()
# print(knn_predictions)


def nb():
    grid = {'alpha': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    model = MultinomialNB()

    predictions = k_cross_validation(model, grid)

    # try:
    #     file = open(str(model) + '_bow', 'rb')
    #     final_model = pickle.load(file)
    #     file.close()
    #
    #     predictions = evaluate(final_model)
    #
    # except (OSError, IOError) as e:
    #     k_cross_validation(model, grid)
    #
    #     file = open(str(model) + '_bow', 'rb')
    #     final_model = pickle.load(file)
    #     file.close()
    #
    #     predictions = evaluate(final_model)

    return predictions


# nb_predictions = nb()
# print(nb_predictions)

def mlp():
    grid = {'hidden_layer_sizes': [(100,), (100,50,)], 'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'], 'alpha': [0.0001, 0.05, 0.1], 'learning_rate': ['constant', 'adaptive'],
            }
    model = MLPClassifier(random_state=42)

    predictions = k_cross_validation(model, grid)

    # try:
    #     file = open(str(model) + '_bow', 'rb')
    #     final_model = pickle.load(file)
    #     file.close()
    #
    #     predictions = evaluate(final_model)
    #
    # except (OSError, IOError) as e:
    #     k_cross_validation(model, grid)
    #
    #     file = open(str(model) + '_bow', 'rb')
    #     final_model = pickle.load(file)
    #     file.close()
    #
    #     predictions = evaluate(final_model)

    return predictions

#
# mlp_predictions = mlp()
# print(mlp_predictions)
# print(mlp_pred_save)


def xgboost():
    grid = {'n_estimators': [10, 100, 200], 'max_depth': [2, 4, 6, 8, 10],
            'learning_rate': [0.01, 0.1, 0.2, 0.3]}
    model = XGBClassifier(random_state=42)


    predictions = k_cross_validation(model, grid)

    # try:
    #     file = open(str(model) + '_bow', 'rb')
    #     final_model = pickle.load(file)
    #     file.close()
    #
    #     predictions = evaluate(final_model)
    #
    # except (OSError, IOError) as e:
    #     k_cross_validation(model, grid)
    #
    #     file = open(str(model) + '_bow', 'rb')
    #     final_model = pickle.load(file)
    #     file.close()
    #
    #     predictions = evaluate(final_model)

    return predictions

# xgboost_predictions = xgboost()
# print(xgboost_predictions)