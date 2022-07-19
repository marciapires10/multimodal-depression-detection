import math
import os
import warnings
from collections import Counter

import keras.utils.np_utils
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn.utils
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from keras.layers import LSTM
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score, recall_score, mean_absolute_error, mean_squared_error
from sklearn.svm import SVC

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Activation, Dropout
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn.utils import shuffle
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD

warnings.filterwarnings('ignore')

df = pd.read_csv("../data_preprocessing/mfcc_features.csv", index_col="Participant_ID")
y = df.PHQ8_Binary

# all features concat with only low variance
df_allfeatures = pd.read_csv("../data_preprocessing/all_audio_features.csv", index_col="Participant_ID",
                             converters={'all_features_concat': pd.eval})

# all features concat with low variance + f1 regression
df_allfeatures2 = pd.read_csv("../data_preprocessing/all_audio_features2.csv", index_col="Participant_ID",
                              converters={'all_features_concat': pd.eval})


X_all = df_allfeatures['all_features_concat'].values.tolist()
X_all2 = df_allfeatures2['all_features_concat'].values.tolist()


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


X_train, X_test, y_train, y_test = train_test(X_all2, y)
X_train, y_train = shuffle(X_train, y_train, random_state=42)

num_labels = 2

# oversampling
def apply_oversampling(X_train, y_train):
    oversample = RandomOverSampler(sampling_strategy='minority')
    X_train, y_train = oversample.fit_resample(X_train, y_train)
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    print(Counter(y_train))

    return X_train, y_train

X_train, y_train = apply_oversampling(X_train, y_train)


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

    #grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"]}
    grid = [{'C': np.logspace(-3, 3, 7), 'penalty': ['l1'], 'solver': ['liblinear', 'sag', 'saga']},
            {'C': np.logspace(-3, 3, 7), 'penalty': ['l2'], 'solver': ['lbfgs', 'newton-cg']}]
    #model = LogisticRegression(n_jobs=3, C=10000)
    model = LogisticRegression()
    f1_score_res, recall_res, mae_res, rmse_res, best_params = k_cross_validation(model, grid)

    f1_val = np.mean(f1_score_res)
    recall_val = np.mean(recall_res)
    mae_val = np.mean(mae_res)
    rmse_val = np.mean(rmse_res)

    print(f"Logistic Regression\nBest F1 Score: {f1_val}\nBest Recall Score: {recall_val}\n"
          f"MAE: {mae_val}\nRMSE: {rmse_val}\n")
    print("Tuned hyperparameters :", best_params)


#logistic_regression()


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

    print(f"Random Forest\nBest F1 Score: {f1_val}\nBest Recall Score: {recall_val}\n"
          f"MAE: {mae_val}\nRMSE: {rmse_val}\n")
    print("Tuned hyperparameters :", best_params)


random_forest()


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


#svm()


# set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)


set_seed(42)

y_train = keras.utils.np_utils.to_categorical(y_train, num_labels)
y_test = keras.utils.np_utils.to_categorical(y_test, num_labels)

def ANN():

    # ANN model
    model = Sequential()
    ### first layer
    model.add(Dense(100, input_shape=(220,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    ### second layer
    model.add(Dense(200))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    ### third layer
    model.add(Dense(100))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    ### final layer
    model.add(Dense(num_labels))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')

    num_epochs = 100
    num_batch_size = 32
    #checkpoint = ModelCheckpoint(filepath='./audio_classification.hdf5', verbose=1, save_best_only=True)
    model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test),
              verbose=1)
    test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(test_accuracy[1])
    print(test_accuracy)

    predict_x = model.predict(X_test)
    classes_x = np.argmax(predict_x, axis=1)
    print(classes_x)

#ANN()

# print(X_train.shape)
#X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1, 1))
#X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1, 1))
# print(X_train.shape)
#
# # CNN
# model = Sequential()
# model.add(Conv2D(80, kernel_size=(3, 3), activation='relu', input_shape=(292, 1, 1, 1)))
# model.add(MaxPooling2D(pool_size=(4, 3)))
# model.add(Conv2D(80, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(1, 3)))
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))
#
# model.summary()
#
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=1, validation_data=(X_test, y_test))
#
# score = model.evaluate(X_test, y_test, verbose=0)
# print("Test loss: ", score[0])
# print("Test accuracy: ", score[1])

# batch_size = 5
# hidden_units = 13
#
# print(len(X_train), 'train sequences')
# print(len(X_test), 'test sequences')
# print('X_train shape:', X_train.shape)
# print('X_test shape:', X_test.shape)
# # print('y_train shape:', y_train.shape)
# # print('y_test shape:', y_test.shape)
# # print(y_test)
# print('Build model...')
#
# X_train = X_train.reshape(1, X_train.shape[0], X_train.shape[1])
# X_test = X_test.reshape(1, X_test.shape[0], X_test.shape[1])
# print('X_train shape:', X_train.shape)
# print('X_test shape:', X_test.shape)
#
# Y_train = keras.utils.np_utils.to_categorical(y_train, num_labels)
# Y_test = keras.utils.np_utils.to_categorical(y_test, num_labels)
# print(Y_train.shape)
# print(Y_test.shape)
# Y_train = Y_train.reshape(1, Y_train.shape[0], Y_train.shape[1])
# Y_test = Y_test.reshape(1, Y_test.shape[0], Y_test.shape[1])
#
# model = Sequential()
# model.add(LSTM(units=hidden_units, kernel_initializer='uniform',
#            unit_forget_bias='one', activation='tanh', recurrent_activation='sigmoid', input_shape=(None,X_train.shape[2]),     return_sequences=True))
#
#
# model.add(Dense(num_labels))
# model.add(Activation('softmax'))
#
# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# print(model.summary())
#
# print("Train...")
# model.fit(X_train, Y_train, batch_size=batch_size, epochs=3)
# score, acc = model.evaluate(X_test, Y_test,
#                         batch_size=batch_size)
# print('Test score:', score)
# print('Test accuracy:', acc)

