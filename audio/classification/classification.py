import math
import os
import warnings
from collections import Counter

import keras.utils.np_utils
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import sklearn.utils
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from keras.layers import LSTM
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score, recall_score, mean_absolute_error, mean_squared_error, \
    classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Activation, Dropout
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn.utils import shuffle
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')

df = pd.read_csv("../data_preprocessing/mfcc_features.csv", index_col="Participant_ID")
y = df.PHQ8_Binary

# all features concat with only low variance
df_allfeatures = pd.read_csv("../data_preprocessing/all_audio_features.csv", index_col="Participant_ID",
                             converters={'all_features_concat': pd.eval})

# all features concat with low variance + f1 regression
# df_allfeatures2 = pd.read_csv("../data_preprocessing/all_audio_features2.csv", index_col="Participant_ID",
#                               converters={'all_features_concat': pd.eval})

# all features
df_allfeatures3 = pd.read_csv("../data_preprocessing/all_audio_features3.csv", index_col="Participant_ID",
                              converters={'all_features_concat': pd.eval})


# only covarep + formant features
df_cov_for = pd.read_csv("../data_preprocessing/covarep_formant_features.csv", index_col="Participant_ID",
                              converters={'all_features_concat': pd.eval})

df_allfeatures4 = pd.read_csv("../data_preprocessing/all_audio_features4.csv", index_col="Participant_ID",
                              converters={'all_features_concat': pd.eval})


X_all = df_allfeatures['all_features_concat'].values.tolist()
# X_all2 = df_allfeatures2['all_features_concat'].values.tolist()
X_all3 = df_allfeatures3['all_features_concat'].values.tolist()
X_all4 = df_cov_for['all_features_concat'].values.tolist()
X_all5 = df_allfeatures4['all_features_concat'].values.tolist()

# set seed for reproducibility
# def set_seed(seed):
#     np.random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     tf.random.set_seed(seed)
#
#
# set_seed(42)

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


X_train, X_test, y_train, y_test = train_test(X_all5, y)
X_train, y_train = shuffle(X_train, y_train, random_state=42)

print(X_train.shape)


# sel = VarianceThreshold(threshold=0.1)
# X_train = sel.fit_transform(X_train, y_train)
# X_test = sel.transform(X_test)
#
#
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
#
chi2_selector = SelectKBest(score_func=f_classif, k=150)
X_train = chi2_selector.fit_transform(X_train, y_train)
X_test = chi2_selector.transform(X_test)


print(X_train.shape)
#
# pca = PCA()
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)

print(X_train.shape)

num_labels = 2


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


X_train, y_train = apply_resampling("smote", X_train, y_train)


# save data sets
# train_x = open('X_train_audio.pickle', 'wb')
# pickle.dump(X_train, train_x)
# train_x.close()
#
# train_y = open('y_train_audio.pickle', 'wb')
# pickle.dump(y_train, train_y)
# train_y.close()
#
# test_x = open('X_test_audio.pickle', 'wb')
# pickle.dump(X_test, test_x)
# test_x.close()
#
# test_y = open('y_test_audio.pickle', 'wb')
# pickle.dump(y_test, test_y)
# test_y.close()


def k_cross_validation(model, grid, X=X_train, y=y_train):
    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    scoring = {"f1": make_scorer(f1_score), "recall": make_scorer(recall_score)}

    cv = GridSearchCV(model, grid, cv=rkf, scoring=scoring, refit='f1', n_jobs=-1)
    # cv = GridSearchCV(model, grid, cv=rkf, refit=True, verbose=3)

    f1_score_res = []
    recall_res = []

    for train_index, val_index in rkf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        cv.fit(X_train, y_train)
        predict_values = cv.predict(X_val)

        f1_scr = f1_score(y_val, predict_values)
        f1_score_res.append(f1_scr)

        recall = recall_score(y_val, predict_values)
        recall_res.append(recall)

    f1_val = np.mean(f1_score_res)
    recall_val = np.mean(recall_res)

    print(f"Training...\nBest F1 Score: {f1_val}\nBest Recall Score: {recall_val}\n")
    print("Tuned hyperparameters: ", cv.best_params_)
    print("best esti ", cv.best_estimator_)

    train_predictions = cv.predict(X_train)
    grid_predictions = cv.predict(X_test)
    print("predictions: ", grid_predictions)

    # print classification report
    # print(classification_report(y_test, grid_predictions))

    f1_scr_train = f1_score(y_train, train_predictions, average='weighted')
    f1_scr = f1_score(y_test, grid_predictions, average='weighted')
    recall = recall_score(y_test, grid_predictions)

    print("Train F1-Score: ", f1_scr_train)
    print("F1-Score: ", f1_scr)
    print("Recall: ", recall)

    # save best model
    # file = open(str(model) + '_audio', 'wb')
    # pickle.dump(cv.best_estimator_, file)
    # file.close()

    predictions = evaluate(cv.best_estimator_)

    return predictions


def evaluate(best_model):

    #grid = GridSearchCV(model, param_grid, refit=True, verbose=3, cv=5, n_jobs=-1)

    # fitting the model for grid search
    #grid.fit(X_train, y_train)

    # print best parameter after tuning
    # print(grid.best_params_)
    #
    # print how our model looks after hyper-parameter tuning
    # print(grid.best_estimator_)
    #
    # print(grid.best_score_)

    train_predictions = best_model.predict(X_train)
    grid_predictions = best_model.predict(X_test)
    print("predictions: ", grid_predictions)

    print(classification_report(y_train, train_predictions))

    # print classification report
    print(classification_report(y_test, grid_predictions))

    f1_scr_train = f1_score(y_train, train_predictions, average='weighted')
    f1_scr = f1_score(y_test, grid_predictions, average='weighted')
    recall = recall_score(y_test, grid_predictions)

    print("Train F1-Score: ", f1_scr_train)
    print("F1-Score: ", f1_scr)
    print("Recall: ", recall)

    return grid_predictions



def logistic_regression():

    # grid = [{'C': np.logspace(-3, 3, 7), 'penalty': ['l1'], 'solver': ['liblinear', 'sag', 'saga']},
    #         {'C': np.logspace(-3, 3, 7), 'penalty': ['l2'], 'solver': ['lbfgs', 'newton-cg']}]
    # model = LogisticRegression(max_iter=30000)

    grid = [{'C': np.logspace(-3, 3, 7), 'penalty': ['l1']}]
    model = LogisticRegression(solver='liblinear', max_iter=10000)

    predictions = k_cross_validation(model, grid)

    # try:
    #     file = open(str(model) + '_audio', 'rb')
    #     final_model = pickle.load(file)
    #     file.close()
    #
    #     predictions = evaluate(final_model)
    #
    # except (OSError, IOError) as e:
    #     k_cross_validation(model, grid)
    #
    #     file = open(str(model) + '_audio', 'rb')
    #     final_model = pickle.load(file)
    #     file.close()
    #
    #     predictions = evaluate(final_model)

    return predictions


lr_predictions = logistic_regression()
print(lr_predictions)


def random_forest():

    grid = {
        'n_estimators': [1, 10, 100, 1000, 2000],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 6, 7, 8],
        'criterion': ['gini', 'entropy']
    }
    model = RandomForestClassifier(random_state=42)

    predictions = k_cross_validation(model, grid)


    return predictions


# rf_predictions = random_forest()
# print(rf_predictions)


def decision_tree():

    grid = {
        'criterion': ['gini', 'entropy'], 'max_depth': [2, 4, 6, 8, 10, 12],
    }
    model = DecisionTreeClassifier(random_state=42, min_samples_leaf=10)

    predictions = k_cross_validation(model, grid)

    return predictions


# dt_predictions = decision_tree()
# print(dt_predictions)


def svm():

    grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100]},
            {'kernel': ['poly'], 'degree': [3, 4, 5], 'C': [1, 10, 100]},
            {'kernel': ['linear'], 'C': [1, 10, 100]}]
    model = SVC(class_weight='balanced')

    predictions = k_cross_validation(model, grid)

    return predictions


# svm_predictions = svm()
# print(svm_predictions)

# pred_final = (lr_predictions + rf_predictions + dt_predictions + svm_predictions) / 4.0
# print(pred_final)
#
#
# def round_up(x):
#     i, f = divmod(x, 1)
#     return int(i + ((f >= 0.5) if (x > 0) else (f > 0.5)))
#
#
# pred = []
# for p in pred_final:
#     p = round_up(p)
#     pred.append(p)
#
# print(pred)
#
# print(f1_score(y_test, pred))
# print(recall_score(y_test, pred))

#saved_model = svm()


#svm_from_pickle = pickle.loads(saved_model)

#pred1=svm_from_pickle.predict(X_test)
#print(pred1)
#print(classification_report(y_test, pred1))
# pred2=model2.predict_proba(x_test)
# pred3=model3.predict_proba(x_test)
#
#
# finalpred=(pred1*0.3+pred2*0.3+pred3*0.4)

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



y_train = keras.utils.np_utils.to_categorical(y_train, num_labels)
y_test = keras.utils.np_utils.to_categorical(y_test, num_labels)

def ANN():

    es = EarlyStopping(patience=20, restore_best_weights=True)

    # ANN model
    model = Sequential()
    ### first layer
    model.add(Dense(200, input_shape=(240,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    ### second layer
    # model.add(Dense(200))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.4))
    ### third layer
    # model.add(Dense(100))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.2))
    ### final layer
    model.add(Dense(num_labels))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')

    num_epochs = 200
    num_batch_size = 16
    model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_split=0.2,
              verbose=1, callbacks=[es])
    test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print("Test accuracy: ", test_accuracy[1])

    predict_x = model.predict(X_test)
    classes_x = np.argmax(predict_x, axis=1)
    print(classes_x)
    y_test2 = np.argmax(y_test, axis=1)


    f1_s = f1_score(y_test2, classes_x)
    print("f1-score: ", f1_s)

    recall = recall_score(y_test2, classes_x)
    print("recall: ", recall)


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


# model_audio = Sequential()
# ### first layer
# model_audio.add(Dense(200, input_shape=(226,)))
# model_audio.add(Activation('relu'))
# model_audio.add(Dropout(0.3))
# ### second layer
# model_audio.add(Dense(200))
# model_audio.add(Activation('relu'))
# #model.add(Dropout(0.5))
# ### third layer
# model_audio.add(Dense(100))
# model_audio.add(Activation('relu'))
# #model.add(Dropout(0.5))
# ### final layer
# model_audio.add(Dense(1))
# model_audio.add(Activation('sigmoid'))
#
# model_audio.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
#
# num_epochs = 10
# num_batch_size = 32
# #checkpoint = ModelCheckpoint(filepath='./audio_classification.hdf5', verbose=1, save_best_only=True)
# model_audio.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs,
#                 validation_data=(X_test, y_test), verbose=1)
#
# acc = model_audio.evaluate(X_test, y_test)
# print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(acc[0], acc[1]))


print(X_train.shape)