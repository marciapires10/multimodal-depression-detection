import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.utils import shuffle

df = pd.read_csv("../full_labels.csv")
x = df.Participant_ID
y = df.PHQ8_Binary

# mean_depressed = []
# mean_not_depressed = []


# for idx, id in enumerate(x):
#     try:
#         if y[idx] == 1:
#             path = "/media/marciapires/My Passport/audio/{}.csv".format(id)
#             df = pd.read_csv(path, header=None)
#             #print(df.values[2])
#             #print(path)
#             #print(df.columns.values)
#             mean_depressed.append(df[0][0])
#         else:
#             path = "/media/marciapires/My Passport/audio/{}.csv".format(id)
#             df = pd.read_csv(path, header=None)
#             mean_not_depressed.append(df[0][0])
#     except:
#         print("Participant " + str(id) + " doesn't exist.")


# mean_depressed = np.mean(mean_depressed)
# print(mean_depressed)
#
# mean_not_depressed = np.mean(mean_not_depressed)
# print(mean_not_depressed)

def get_low_variance_features():

    all_mean = []
    for idx, id in enumerate(x):
        try:
            path = "/media/marciapires/My Passport/audio/{}_cov.csv".format(id)
            df = pd.read_csv(path, header=None)

            #print(df.iloc[0])
            # remove infinite values
            df.replace([np.inf, -np.inf], np.nan, inplace=True)

            # add mean row to the array
            all_mean.append(df.iloc[0].tolist())
        except:
            print("Participant " + str(id) + " doesn't exist.")

    sel = VarianceThreshold(threshold=0.1)
    high_variance = sel.fit_transform(all_mean)

    print(high_variance.shape)
    print(sel.get_support())


get_low_variance_features()


def remove_low_variance_features():
    for idx, id in enumerate(x):
        try:
            path = "/media/marciapires/My Passport/audio/{}_cov.csv".format(id)
            df = pd.read_csv(path, header=None)
            #new_df = df.drop(
                #df.columns[[1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                #            27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 61, 62,
                #            63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73]], axis=1)

            # with only voiced segments, two more features were added
            new_df = df.drop(
                df.columns[[1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                            27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 61, 62,
                            63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73
                            ]], axis=1
            )

            new_path = "/media/marciapires/My Passport/audio/{}_ftSelection.csv".format(id)
            new_df.to_csv(new_path, index=False, header=None)

        except:
            print("Participant " + str(id) + " doesn't exist.")


remove_low_variance_features()


def get_f1_regression_features():
    all_mean_feat = []
    for idx, id in enumerate(x):
        try:
            path = "/media/marciapires/My Passport/audio/{}_ftSelection.csv".format(id)
            df = pd.read_csv(path, header=None)

            all_mean_feat.append(df.iloc[0].tolist())

        except:
            print("Participant " + str(id) + " doesn't exist.")

    X_covarep = all_mean_feat
    df = pd.read_csv("mfcc_features.csv", index_col="Participant_ID")
    y_covarep = df.PHQ8_Binary

    def train_test(X, y, test="../text/test_split_Depression_AVEC2017.csv"):
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

    X_train, X_test, y_train, y_test = train_test(X_covarep, y_covarep)
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    # feature selection
    f_selector = SelectKBest(score_func=f_regression, k='all')
    f_selector.fit(X_train, y_train)
    plt.bar([i for i in range(len(f_selector.scores_))], f_selector.scores_)

    plt.xlabel("feature index")
    plt.ylabel("F-value (transformed from the correlation values)")
    plt.show()

    print(f_selector.scores_)


get_f1_regression_features()


def remove_f1_regression_features():

    for idx, id in enumerate(x):
        try:
            path = "/media/marciapires/My Passport/audio/{}_ftSelection.csv".format(id)
            df = pd.read_csv(path, header=None)
            # new_df = df.drop(
            #     df.columns[[2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16]], axis=1)

            # with only voiced segments
            new_df = df.drop(
                df.columns[[2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18]], axis=1)


            new_path = "/media/marciapires/My Passport/audio/{}_ftSelection2.csv".format(id)
            new_df.to_csv(new_path, index=False, header=None)

        except:
            print("Participant " + str(id) + " doesn't exist.")


remove_f1_regression_features()

