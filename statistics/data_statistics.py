import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../text/data_preprocessing/final_clean_transcripts5.csv").sort_values(by=['Participant_ID'], ascending=True)

# # histogram with PHQ8 Score distribution
# plt.hist(df.PHQ8_Score, bins=24)
# plt.title('PHQ8 Score Distribution (n={})'.format(len(df)))
# plt.ylabel('Distribution')
# plt.xlabel('PHQ8 Score')
# plt.show()
#

###################################################################################################
# distribution of depression presence
# x_labels = ["Not depressed", "Depressed"]
# y_val = [round(len(df[df.PHQ8_Binary == 0])), round(len(df[df.PHQ8_Binary == 1]))]
# p = plt.bar(x_labels, y_val)
# plt.bar_label(p)
# c = ['#B9E0A5', '#CDA2BE']
# plt.bar(x_labels, y_val, align='center', color=c)
# plt.title('Depression presence (n total={})'.format(len(df)))
# plt.ylabel('Number of participants')
# #plt.xlabel('Presence of depression')
# plt.xticks(x_labels)
# plt.savefig('depression_presence2.jpg')
# plt.show()


###################################################################################################

# dataset distribution per gender
# x_labels = ["Female", "Male"]
# y_val = [round(len(df[df.Gender == 0])), round(len(df[df.Gender == 1]))]
# c = ['#B9E0A5', '#CDA2BE']
# plt.bar(x_labels, y_val, align='center')
# plt.title('Gender distribution (n total={})'.format(len(df)))
# plt.ylabel('Number of participants participants')
# plt.xlabel('Gender')
# plt.xticks(x_labels)
# plt.savefig('gender2.jpg')
# plt.show()


###################################################################################################

# histogram with PHQ8 Score distribution (train + test)
X = df["Participant_ID"].values
# y = df["PHQ8_Score"].values
#
#
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

def train_test_val(X, y, test="/home/marciapires/Desktop/multimodal-depression-detection/test_split_Depression_AVEC2017.csv",
                   val="/home/marciapires/Desktop/multimodal-depression-detection/dev_split_Depression_AVEC2017.csv"):
    test_participants = pd.read_csv(test)['participant_ID'].values
    val_participants = pd.read_csv(val)['Participant_ID'].values
    X_train = []
    X_test = []
    X_val = []
    y_train = []
    y_test = []
    y_val = []

    for i in range(y.shape[0]):
        participant_no = y.index[i]

        if participant_no in test_participants:
            X_test.append(X[i])
            y_test.append(y[participant_no])
        elif participant_no in val_participants:
            X_val.append(X[i])
            y_val.append(y[participant_no])
        else:
            X_train.append(X[i])
            y_train.append(y[participant_no])

    return np.array(X_train), np.array(X_test), np.array(X_val), np.array(y_train), np.array(y_test), np.array(y_val)



y = pd.read_csv("../text/data_preprocessing/final_clean_transcripts5.csv", index_col="Participant_ID")['PHQ8_Score']

X_train, X_test, X_val, y_train, y_test, y_val = train_test_val(X, y)
c1 = ['#B9E0A5']
c2 = ['#CDA2BE']
c3 = ['#FFCE9F']
plt.hist(y_test, bins=24, color=c3)
# plt.hist(y_test, bins=24, label="test")
plt.xticks(np.arange(0, 25, 4.0))
plt.title('Test set PHQ-8 Score Distribution (n total={})'.format(len(y_test)))
plt.ylabel('Number of participants')
plt.xlabel('PHQ-8 Score')
#plt.legend()
plt.savefig('score_distribution_test.jpg')
plt.show()


####################################################################################################

# distribution of depression presence (train + test)

y = pd.read_csv("../text/data_preprocessing/final_clean_transcripts5.csv", index_col="Participant_ID")['PHQ8_Binary']
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


#x_labels = ["Not depressed", "Depressed"]
#x_labels = ["Train", "Test"]

# y_train_count = count_depression(y_train)
# y_test_count = count_depression(y_test)
# c1 = ['#B9E0A5']
# c2 = ['#CDA2BE']
# X_axis = np.arange(len(x_labels))
# plt.xticks(X_axis, x_labels)
# bar1 = plt.bar(X_axis - 0.2, [y_train_count[0], y_test_count[0]], align='center', label="Not depressed", color=c1, width=0.4)
# bar2 = plt.bar(X_axis + 0.2, [y_train_count[1], y_test_count[1]], align='center', label="Depressed", color=c2, width=0.4)
# plt.title('Depression presence (n total={})'.format(len(df)))
# plt.ylabel('Number of participants')
# plt.bar_label(bar1)
# plt.bar_label(bar2)
# plt.legend()
# plt.savefig('depression_presence_tt2.jpg')
# plt.show()


####################################################################################################

# distribution of depression presence (train + val + test)
df = pd.read_csv("../text/data_preprocessing/final_clean_transcripts5.csv").sort_values(by=['Participant_ID'], ascending=True)
X = df["Participant_ID"].values
y = df['PHQ8_Binary'].values


def train_test_val(X, y, test="/home/marciapires/Desktop/multimodal-depression-detection/test_split_Depression_AVEC2017.csv",
                   val="/home/marciapires/Desktop/multimodal-depression-detection/dev_split_Depression_AVEC2017.csv"):
    test_participants = pd.read_csv(test)['participant_ID'].values
    val_participants = pd.read_csv(val)['Participant_ID'].values
    X_train = []
    X_test = []
    X_val = []
    y_train = []
    y_test = []
    y_val = []

    for i in range(y.shape[0]):
        participant_no = y.index[i]

        if participant_no in test_participants:
            X_test.append(X[i])
            y_test.append(y[participant_no])
        elif participant_no in val_participants:
            X_val.append(X[i])
            y_val.append(y[participant_no])
        else:
            X_train.append(X[i])
            y_train.append(y[participant_no])

    return np.array(X_train), np.array(X_test), np.array(X_val), np.array(y_train), np.array(y_test), np.array(y_val)


y = pd.read_csv("../text/data_preprocessing/final_clean_transcripts5.csv", index_col="Participant_ID")['PHQ8_Binary']
X_train, X_test, X_val, y_train, y_test, y_val = train_test_val(X, y)

# x_labels = ['Train', 'Dev', 'Test']
#
# y_train_count = count_depression(y_train)
# y_val_count = count_depression(y_val)
# y_test_count = count_depression(y_test)
# c1 = ['#B9E0A5']
# c2 = ['#CDA2BE']
# c3 = ['#FFE599']
# # p = plt.bar(x_labels, y_train_count)
# # plt.bar_label(p)
# # p2 = plt.bar(x_labels, y_val_count)
# # plt.bar_label(p2)
# # p3 = plt.bar(x_labels, y_test_count)
# # plt.bar_label(p3)
# X_axis = np.arange(len(x_labels))
# plt.xticks(X_axis, x_labels)
# bar1 = plt.bar(X_axis - 0.2, [y_train_count[0], y_val_count[0], y_test_count[0]], align='center', label="Not depressed", color=c1, width=0.4)
# bar2 = plt.bar(X_axis + 0.2, [y_train_count[1], y_val_count[1], y_test_count[1]], align='center', label="Depressed", color=c2, width=0.4)
# #plt.bar(x_labels, y_test_count, align='center', label="test", color=c3)
# plt.title('Depression presence (n total={})'.format(len(df)))
# plt.ylabel('Number of participants')
# plt.bar_label(bar1)
# plt.bar_label(bar2)
# plt.legend()
# #plt.tight_layout()
# plt.savefig('depression_presence_tdt.jpg')
# plt.show()


# y = pd.read_csv("../data_preprocessing/final_clean_transcripts5.csv", index_col="Participant_ID")['PHQ8_Score']
# X_train, X_test, X_val, y_train, y_test, y_val = train_test_val(X, y)
#
#
# c1 = ['#B9E0A5']
# c2 = ['#CDA2BE']
# c3 = ['#FFE599']
# plt.hist(y_train, bins=24, label="train", color=c1)
# plt.hist(y_val, bins=24, label="dev", color=c2)
# plt.hist(y_test, bins=24, label="test", color=c3)
# plt.title('PHQ-8 Score Distribution (n total={})'.format(len(df)))
# plt.ylabel('Distribution')
# plt.xlabel('PHQ-8 Score')
# plt.legend()
# plt.savefig('score_distribution_tdt.jpg')
# plt.show()