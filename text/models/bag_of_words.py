import math
import random
import warnings
from collections import Counter
from math import floor, log

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from keras import Sequential
from keras.layers import Embedding, Dropout, Conv1D, MaxPooling1D, Flatten, Dense, LSTM
from keras_preprocessing import text, sequence
from numpy import asarray, zeros
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, make_scorer, classification_report, accuracy_score, \
    mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle, class_weight

warnings.filterwarnings('ignore')

df = pd.read_csv("../data_preprocessing/final_clean_transcripts.csv", index_col="Participant_ID")
transcripts = df.Transcript

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
            X_test.append(X[participant_no])
            y_test.append(y[participant_no])
        else:
            X_train.append(X[participant_no])
            y_train.append(y[participant_no])

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


X_train, X_test, y_train, y_test = train_test(transcripts, y)
X_train, y_train = shuffle(X_train, y_train, random_state=42)
print(Counter(y_train))

X_train_bow = cv.fit_transform(X_train).toarray()
X_test_bow = cv.transform(X_test).toarray()

print(X_test_bow)

print(X_train_bow.shape)
print(X_test_bow.shape)

# def train_test_val(X, y, test="../test_split_Depression_AVEC2017.csv", val="../dev_split_Depression_AVEC2017.csv"):
#     test_participants = pd.read_csv(test)['participant_ID'].values
#     val_participants = pd.read_csv(val)['Participant_ID'].values
#     X_train = []
#     X_test = []
#     X_val = []
#     y_train = []
#     y_test = []
#     y_val = []
#
#     for i in range(y.shape[0]):
#         participant_no = y.index[i]
#
#         if participant_no in test_participants:
#             X_test.append(X[participant_no])
#             y_test.append(y[participant_no])
#         # elif participant_no in val_participants:
#         #    X_val.append(X[participant_no])
#         #    y_val.append(y[participant_no])
#         else:
#             X_train.append(X[participant_no])
#             y_train.append(y[participant_no])
#
#     return np.array(X_train), np.array(X_test), np.array(X_val), np.array(y_train), np.array(y_test), np.array(y_val)
#
#
# X_train, X_test, X_val, y_train, y_test, y_val = train_test_val(transcripts, y)

#X_train = X_train.reshape(-1, 1)
# oversample = RandomOverSampler(sampling_strategy='minority')
#X_train_bow = X_train_bow.reshape(-1, 1)
undersample = RandomUnderSampler(sampling_strategy='majority')
X_train_bow, y_train = undersample.fit_resample(X_train_bow, y_train)
print(Counter(y_train))
X_train_bow, y_train = shuffle(X_train_bow, y_train, random_state=42)


def k_cross_validation(model, grid, X=X_train_bow, y=y_train):
    rkf = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)
    scoring = {"f1": make_scorer(f1_score), "recall": make_scorer(recall_score)}

    cv = GridSearchCV(model, grid, cv=rkf, scoring=scoring, refit='f1')

    f1_score_res = []
    recall_res = []
    mae_res = []
    rmse_res = []

    for train_index, test_index in rkf.split(X):
        X_train_bow, X_test_bow = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        cv.fit(X_train_bow, y_train)
        predict_values = cv.predict(X_test_bow)

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
    #grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l1"], "solver": ['liblinear']}
    grid = [{'C': np.logspace(-3, 3, 7), 'penalty': ['l1'], 'solver': ['liblinear', 'sag', 'saga']},
            {'C': np.logspace(-3, 3, 7), 'penalty': ['l2'], 'solver': ['lbfgs', 'newton-cg']}]
    # model = LogisticRegression(n_jobs=3, C=10000)
    model = LogisticRegression()
    f1_score_res, recall_res, mae_res, rmse_res, best_params = k_cross_validation(model, grid)

    f1_val = np.mean(f1_score_res)
    recall_val = np.mean(recall_res)
    mae_val = np.mean(mae_res)
    rmse_val = np.mean(rmse_res)

    print(f"Logistic Regression\nBest F1 Score: {f1_val}\nBest Recall Score: {recall_val}\nMAE: {mae_val}\nRMSE: {rmse_val}\n")
    print("Tuned hyperparameters :", best_params)


# logistic_regression()


def random_forest():
    grid = {
        'n_estimators': [1, 10],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 6, 7, 8],
        'criterion': ['gini', 'entropy']
    }
    model = RandomForestClassifier(random_state=42)
    f1_score_res, recall_res, mae_res, rmse_res, best_params = k_cross_validation(model, grid)

    f1_val = np.mean(f1_score_res)
    recall_val = np.mean(recall_res)
    mae_val = np.mean(mae_res)
    rmse_val = np.mean(rmse_res)

    print(f"Random Forest\nBest F1 Score: {f1_val}\nBest Recall Score: {recall_val}\nMAE: {mae_val}\nRMSE: {rmse_val}\n")
    print("Tuned hyperparameters :", best_params)


# random_forest()


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

    print(f"Decision Tree\nBest F1 Score: {f1_val}\nBest Recall Score: {recall_val}\nMAE: {mae_val}\nRMSE: {rmse_val}\n")
    print("Tuned hyperparameters :", best_params)


# decision_tree()


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

    print(f"SVM \nBest F1 Score: {f1_val}\nBest Recall Score: {recall_val}\nMAE: {mae_val}\nRMSE: {rmse_val}\n")
    print("Tuned hyperparameters :", best_params)


# svm()


def multinomialNB():
    grid = {}
    model = MultinomialNB()

    f1_score_res, recall_res, mae_res, rmse_res, best_params = k_cross_validation(model, grid)

    f1_val = np.mean(f1_score_res)
    recall_val = np.mean(recall_res)
    mae_val = np.mean(mae_res)
    rmse_val = np.mean(rmse_res)

    print(f"Multinomial Naive Bayes \nBest F1 Score: {f1_val}\nBest Recall Score: {recall_val}\nMAE: {mae_val}\nRMSE: {rmse_val}\n")
    print("Tuned hyperparameters :", best_params)


#multinomialNB()

def mlp():
    grid = {'solver': ['adam']}
    model = MLPClassifier(verbose=True, max_iter=300)

    f1_score_res, recall_res, mae_res, rmse_res, best_params = k_cross_validation(model, grid)

    f1_val = np.mean(f1_score_res)
    recall_val = np.mean(recall_res)
    mae_val = np.mean(mae_res)
    rmse_val = np.mean(rmse_res)

    print(f"MLP \nBest F1 Score: {f1_val}\nBest Recall Score: {recall_val}\nMAE: {mae_val}\nRMSE: {rmse_val}\n")
    print("Tuned hyperparameters :", best_params)


#mlp()

vocab_size = 0
embed_size = 300
epochs = 10
batch_size = 128

words_per_transcript = list(map(lambda x: len(x.split()), transcripts))
avg = sum(words_per_transcript) / len(words_per_transcript)
avg = pow(2, floor(log(avg) / log(2)))

results = Counter()
transcripts.str.lower().str.split().apply(results.update)

max_length_content = int(avg)
vocab_size = len(results) + 1

token = text.Tokenizer(lower=False, num_words=vocab_size)
token.fit_on_texts(transcripts)
word_index = token.word_index

X_train = X_train.ravel()
#Xcnn_train = sequence.pad_sequences(token.texts_to_sequences(X_train), maxlen=max_length_content)
#Xcnn_test = sequence.pad_sequences(token.texts_to_sequences(X_test), maxlen=max_length_content)


## word embeddings with glove
def load_embedding(filename):
    # load embedding into memory, skip first line
    file = open(filename, 'r')
    lines = file.readlines()
    file.close()
    # create a map of words to vectors
    embedding = dict()
    for line in lines:
        parts = line.split()
        # key is string word, value is numpy array for vector
        embedding[parts[0]] = asarray(parts[1:], dtype='float32')
    return embedding


# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab):
    # total vocabulary size plus 0 for unknown words
    vocab_size = len(vocab) + 1
    # define weight matrix dimensions with all 0
    weight_matrix = zeros((vocab_size, 300))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in vocab.items():
        vector = embedding.get(word)
        if vector is not None:
            weight_matrix[i] = vector
    return weight_matrix


raw_embedding = load_embedding('../glove.6B.300d.txt')

# get vectors in the right order
embedding_vectors = get_weight_matrix(raw_embedding, word_index)
# create the embedding layer
embedding_layer = Embedding(vocab_size, 300, weights=[embedding_vectors], input_length=max_length_content, trainable=False)

#
# def cnn():
#
#     ### new 02.06 ###
#     model = Sequential()
#     model.add(embedding_layer)
#     model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
#     model.add(MaxPooling1D(pool_size=2))
#     model.add(Flatten())
#     model.add(Dense(1, activation='sigmoid'))
#     print(model.summary())
#     # compile network
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    ### good ###

    # model = Sequential()
    #
    # model.add(Embedding(vocab_size, embed_size, input_length=max_length_content))
    # hp_rates = hp.Choice('rate', values=[0.3, 0.4, 0.5])
    # model.add(Dropout(rate=hp_rates))
    # hp_filters = hp.Int('filters', min_value=10, max_value=128)
    # hp_kernels = hp.Int('kernel_size', min_value=2, max_value=10)
    # model.add(Conv1D(filters=hp_filters, kernel_size=hp_kernels, padding='valid', activation='relu'))
    # model.add(MaxPooling1D())
    # # model.add(Conv1D(filters=hp_filters, kernel_size=hp_kernels, padding='valid', activation='relu'))
    # # model.add(MaxPooling1D())
    # model.add(Flatten())
    # hp_units = hp.Int('units', min_value=10, max_value=100, step=5)
    # model.add(Dense(units=hp_units, activation='relu'))
    # model.add(Dropout(rate=hp_rates))
    # model.add(Dense(1, activation='sigmoid'))

    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    ### good ###
    # model.summary()

    # class_weights = class_weight.compute_class_weight(class_weight='balanced',
    #                                                  classes=np.unique(y_train),
    #                                                  y=y_train)

    # return model

    # scores = model.evaluate(we_test, y_test, verbose=1)
    # print("Accuracy: ", (scores[1] * 100))

    # predictions = np.argmax(model.predict(we_test), axis=-1)
    # print(predictions)

    # print(classification_report(y_test, predictions))


# model = cnn()
# # fit network
# model.fit(Xcnn_train, y_train, epochs=10, verbose=2)
# # evaluate
# loss, acc = model.evaluate(Xcnn_test, y_test, verbose=0)
# print('Test Accuracy: %f' % (acc * 100))



# model = cnn()
# model.fit(we_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, verbose=1)
#
# predictions = model.predict(Xcnn_test)
# predictions = [int(round(p[0])) for p in predictions]
#
# print(predictions)
# print(y_test)
# accuracy = accuracy_score(y_test, predictions) * 100
# print("Accuracy: ", accuracy)
#
# print(classification_report(y_test, predictions))

### good ###
# from kerastuner import RandomSearch
#
# tuner = RandomSearch(cnn, objective='val_accuracy', max_trials=5)
#
# # search best parameter
# tuner.search(Xcnn_train, y_train, epochs=3, validation_split=0.1)
#
# model = tuner.get_best_models(num_models=1)[0]
# #summary of best model
# model.summary()
#
# model.fit(Xcnn_train, y_train, epochs=10, validation_data=(Xcnn_test, y_test), batch_size=32, verbose=2)
#
#
# predictions = model.predict(Xcnn_test)
# predictions = [int(round(p[0])) for p in predictions]
# print(predictions)
#
# loss, accuracy = model.evaluate(Xcnn_train, y_train, verbose=0)
# print("Training Accuracy: {:.4f}".format(accuracy))
#
# loss, accuracy = model.evaluate(Xcnn_test, y_test, verbose=0)
# print('Test Accuracy: {:.4f}'.format(accuracy))

### good ###


# predictions = model.predict(we_test)
# print(predictions)
# predictions = [int(round(p[0])) for p in predictions]
#
# print(predictions)
# print(y_test)
# accuracy = accuracy_score(y_test, predictions) * 100
# print("Accuracy: ", accuracy)
#
# print(classification_report(y_test, predictions))


#### LSTM ####

# def lstm():
#     model = Sequential()
#     model.add(Embedding(vocab_size, embed_size, weights=[embedding_matrix], input_length=max_length_content))
#     model.add(LSTM(units=128))
#     model.add(Dense(1, activation='sigmoid'))
#
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model

# model = lstm()
# model.fit(Xcnn_train, y_train, batch_size=32, epochs=10)
#
# predictions = model.predict(Xcnn_test)
# predictions = [int(round(p[0])) for p in predictions]
#
# print(predictions)
# print(y_test)
#
# loss, accuracy = model.evaluate(Xcnn_train, y_train, verbose=0)
# print("Training Accuracy: {:.4f}".format(accuracy))
#
# loss, accuracy = model.evaluate(Xcnn_test, y_test, verbose=0)
# print('Test Accuracy: {:.4f}'.format(accuracy))
