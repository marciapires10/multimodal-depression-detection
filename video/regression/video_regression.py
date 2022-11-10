import math
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.utils import shuffle

warnings.filterwarnings('ignore')

df = pd.read_csv("../../text/data_preprocessing/final_clean_transcripts5.csv", index_col="Participant_ID")

df_allfeatures = pd.read_csv("/home/marciapires/Desktop/multimodal-depression-detection/video/data_preprocessing/all_video_features2.csv", index_col="Participant_ID",
                             converters={'all_features_concat': pd.eval})



X_all = df_allfeatures['all_features_concat'].values.tolist()
y = df.PHQ8_Score

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


X_train, X_test, y_train, y_test = train_test(X_all, y)
X_train, y_train = shuffle(X_train, y_train, random_state=42)


# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

reg_selector = SelectKBest(score_func=f_regression, k=250)
X_train = reg_selector.fit_transform(X_train, y_train)
X_test = reg_selector.transform(X_test)


# save data sets
# train_x = open('X_train_video_r.pickle', 'wb')
# pickle.dump(X_train, train_x)
# train_x.close()
#
# train_y = open('y_train_video_r.pickle', 'wb')
# pickle.dump(y_train, train_y)
# train_y.close()
#
# test_x = open('X_test_video_r.pickle', 'wb')
# pickle.dump(X_test, test_x)
# test_x.close()
#
# test_y = open('y_test_video_r.pickle', 'wb')
# pickle.dump(y_test, test_y)
# test_y.close()

def k_cross_validation(model, grid, X=X_train, y=y_train):
    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

    cv = GridSearchCV(model, grid, cv=rkf, refit=True, n_jobs=-1)
    # cv = GridSearchCV(model, grid, cv=rkf, refit=True, verbose=3)

    mae_res = []
    rmse_res = []

    for train_index, val_index in rkf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        cv.fit(X_train, y_train)
        predict_values = cv.predict(X_val)

        mae = mean_absolute_error(y_val, predict_values)
        mae_res.append(mae)
        mse = mean_squared_error(y_val, predict_values)
        rmse = math.sqrt(mse)
        rmse_res.append(rmse)

    mae_val = np.mean(mae_res)
    rmse_val = np.mean(rmse_res)

    print(f"Training...\nBest MAE: {mae_val}\nBest RMSE: {rmse_val}\n")
    print("Tuned hyperparameters: ", cv.best_params_)
    print("Best estimator: ", cv.best_estimator_)

    grid_predictions = cv.predict(X_test)
    print("predictions: ", grid_predictions)

    mae = mean_absolute_error(y_test, grid_predictions)
    mse = mean_squared_error(y_test, grid_predictions)
    rmse = math.sqrt(mse)

    print("MAE: ", mae)
    print("RMSE: ", rmse)

    # save best model
    # file = open(str(model) + '_video_r', 'wb')
    # pickle.dump(cv.best_estimator_, file)
    # file.close()

    predictions = evaluate(cv.best_estimator_)

    return predictions


def evaluate(best_model):


    grid_predictions = best_model.predict(X_test)
    print("predictions: ", grid_predictions)
    print("test predictions: ", y_test)

    mae = mean_absolute_error(y_test, grid_predictions)
    mse = mean_squared_error(y_test, grid_predictions)
    rmse = math.sqrt(mse)

    print("MAE: ", mae)
    print("RMSE: ", rmse)

    return grid_predictions


def random_forest():
    grid = {
        'n_estimators': [1, 10, 50, 100],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 6, 7, 8, 9, 10]
    }
    model = RandomForestRegressor(random_state=42)

    predictions = k_cross_validation(model, grid)

    # try:
    #     file = open(str(model) + '_video_r', 'rb')
    #     final_model = pickle.load(file)
    #     file.close()
    #
    #     predictions = evaluate(final_model)
    #
    # except (OSError, IOError) as e:
    #     k_cross_validation(model, grid)
    #
    #     file = open(str(model) + '_video_r', 'rb')
    #     final_model = pickle.load(file)
    #     file.close()
    #
    #     predictions = evaluate(final_model)

    return predictions

# rf_predictions = random_forest()
# print(rf_predictions)

def svr():

    # grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100]},
    #         {'kernel': ['poly'], 'degree': [3, 4, 5], 'C': [1, 10, 100]},
    #         {'kernel': ['linear'], 'C': [1, 10, 100]}]
    grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 0.01, 0.1, 1], 'C': [1, 10, 100], 'epsilon': [1e-3, 0.1, 0.2, 0.3, 0.5, 1]},
            {'kernel': ['poly'], 'C': [1, 10, 100], 'epsilon': [1e-3, 0.1, 0.2, 0.3, 0.5, 1]},
            {'kernel': ['linear'], 'C': [1, 10, 100], 'epsilon': [1e-3, 0.1, 0.2, 0.3, 0.5, 1]}]
    model = SVR()

    predictions = k_cross_validation(model, grid)

    # try:
    #     file = open(str(model) + '_video_r', 'rb')
    #     final_model = pickle.load(file)
    #     file.close()
    #
    #     predictions = evaluate(final_model)
    #
    # except (OSError, IOError) as e:
    #     k_cross_validation(model, grid)
    #
    #     file = open(str(model) + '_video_r', 'rb')
    #     final_model = pickle.load(file)
    #     file.close()
    #
    #     predictions = evaluate(final_model)

    return predictions


# svr_predictions = svr()
# print(svr_predictions)
