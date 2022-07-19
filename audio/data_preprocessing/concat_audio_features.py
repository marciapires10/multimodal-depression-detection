import numpy as np
import pandas as pd

# df = pd.read_csv("full_labels.csv")
# X = df.Participant_ID

### utils

# remove brackets from csv values
def remove_brackets(X):
    wo_brackets = X[1:-1]
    return wo_brackets


# adjust string to stack
def adjust_x(X):
    mfcc_stack = []
    for i in range(len(X)):
        mfcc = np.fromstring(X.iloc[i], dtype=float, sep='\n')
        mfcc_stack.append(mfcc)
    return np.stack(mfcc_stack, axis=0)


# mfcc features
df_mfcc = pd.read_csv("../data_preprocessing/mfcc_features.csv")
X = df_mfcc.Participant_ID
mfcc_ft = df_mfcc.MFCC_concat.apply(remove_brackets)
mfcc_ft = adjust_x(mfcc_ft)

# get covarep or formant features
def get_all_features(source):
    all_features = []
    for idx, id in enumerate(X):
        try:
            if source == "covarep":
                path = "/media/marciapires/My Passport/audio/{}_ftSelection2.csv".format(id)
            else:
                path = "/media/marciapires/My Passport/audio/{}_for.csv".format(id)

            df_audio = pd.read_csv(path, header=None)
            first_row = df_audio.iloc[0].tolist()
            second_row = df_audio.iloc[1].tolist()
            third_row = df_audio.iloc[2].tolist()
            fourth_row = df_audio.iloc[3].tolist()
            fifth_row = df_audio.iloc[4].tolist()
            sixth_row = df_audio.iloc[5].tolist()

            full_features = first_row + second_row + third_row + fourth_row + fifth_row + sixth_row
            all_features.append(full_features)

        except:
            print("Participant " + str(id) + " doesn't exist.")

    return all_features

# covarep features
covarep_ft = get_all_features("covarep")

# formant features
formant_ft = get_all_features("formant")

first_ft = []
all_features = []
features = []

for idx, id in enumerate(X):
    mfcc_covarep = [y for x in [mfcc_ft[idx], covarep_ft[idx]] for y in x]
    first_ft.append(mfcc_covarep)
    prev_formant = [y for x in [first_ft[idx], formant_ft[idx]] for y in x]
    features.append([id, prev_formant])

    new_df = pd.DataFrame(features, columns=['Participant_ID', 'all_features_concat'])
    new_df.to_csv("all_audio_features2.csv", index=False)



