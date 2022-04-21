import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data_preprocessing/final_clean_transcripts.csv").sort_values(by=['Participant_ID'], ascending=True)

# # histogram with PHQ8 Score distribution
# plt.hist(df.PHQ8_Score, bins=24)
# plt.title('PHQ8 Score Distribution (n={})'.format(len(df)))
# plt.ylabel('Distribution')
# plt.xlabel('PHQ8 Score')
# plt.show()
#

###################################################################################################
# distribution of depression presence
x_labels = ["Not depressed", "Depressed"]
y_val = [round(len(df[df.PHQ8_Binary == 0])), round(len(df[df.PHQ8_Binary == 1]))]
plt.bar(x_labels, y_val, align='center')
plt.title('Depression presence (n total={})'.format(len(df)))
plt.ylabel('N participants')
plt.xlabel('Presence of depression')
plt.xticks(x_labels)
plt.savefig('depression_presence.jpg')
plt.show()


###################################################################################################

# dataset distribution per gender
x_labels = ["Female", "Male"]
y_val = [round(len(df[df.Gender == 0])), round(len(df[df.Gender == 1]))]
plt.bar(x_labels, y_val, align='center')
plt.title('Gender distribution (n total={})'.format(len(df)))
plt.ylabel('N participants')
plt.xlabel('Gender')
plt.xticks(x_labels)
plt.savefig('gender.jpg')
plt.show()


###################################################################################################

# histogram with PHQ8 Score distribution (train + test)
X = df["Participant_ID"].values
y = df["PHQ8_Score"].values


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


y = pd.read_csv("../data_preprocessing/final_clean_transcripts.csv", index_col="Participant_ID")['PHQ8_Score']

X_train, X_test, y_train, y_test = train_test(X, y)
plt.hist(y_train, bins=24, label="train")
plt.hist(y_test, bins=24, label="test")
plt.title('PHQ8 Score Distribution (n total={})'.format(len(df)))
plt.ylabel('Distribution')
plt.xlabel('PHQ8 Score')
plt.legend()
plt.savefig('score_distribution.jpg')
plt.show()


####################################################################################################

# distribution of depression presence (train + test)

y = pd.read_csv("../data_preprocessing/final_clean_transcripts.csv", index_col="Participant_ID")['PHQ8_Binary']
X_train, X_test, y_train, y_test = train_test(X, y)


def count_depression(set):
    n_depressed = 0
    depressed = 0
    final = []
    for s in set:
        if s == 0:
            n_depressed += 1
        else:
            depressed += 1

    final.append(n_depressed)
    final.append(depressed)

    return final


x_labels = ["Not depressed", "Depressed"]

y_train_count = count_depression(y_train)
y_test_count = count_depression(y_test)
plt.bar(x_labels, y_train_count, align='center', label="train")
plt.bar(x_labels, y_test_count, align='center', label="test")
plt.title('Depression presence (n total={})'.format(len(df)))
plt.ylabel('N participants')
plt.xticks(x_labels)
plt.legend()
plt.savefig('depression_presence_tt.jpg')
plt.show()