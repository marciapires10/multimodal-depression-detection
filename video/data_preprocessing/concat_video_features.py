import numpy as np
import pandas as pd


# df = pd.read_csv("full_labels.csv")
# X = df.Participant_ID

### utils

# remove brackets from csv values
def remove_brackets(X):
    wo_brackets = X[1:-1]
    return wo_brackets


# mfcc features
# df_mfcc = pd.read_csv("../data_preprocessing/mfcc_features.csv")
# X = df_mfcc.Participant_ID

df = pd.read_csv("/home/marciapires/Desktop/multimodal-depression-detection/audio/data_preprocessing/mfcc_features.csv")
X = df.Participant_ID


# get aus, gaze or/and pose features
def get_all_features(source):
    all_features = []
    for idx, id in enumerate(X):
        try:
            if source == "aus":
                path = "/media/marciapires/My Passport/video/{}_aus.csv".format(id)
            elif source == "gaze":
                path = "/media/marciapires/My Passport/video/{}_gaze.csv".format(id)
            else:
                path = "/media/marciapires/My Passport/video/{}_pose.csv".format(id)

            df_video = pd.read_csv(path, header=None)
            first_row = df_video.iloc[0].tolist()
            second_row = df_video.iloc[1].tolist()
            third_row = df_video.iloc[2].tolist()
            fourth_row = df_video.iloc[3].tolist()
            fifth_row = df_video.iloc[4].tolist()
            sixth_row = df_video.iloc[5].tolist()
            seventh_row = df_video.iloc[6].tolist()

            full_features = first_row + second_row + third_row + fourth_row + fifth_row + sixth_row + seventh_row
            all_features.append(full_features)

        except:
            print("Participant " + str(id) + " doesn't exist.")

    return all_features


# aus features
aus_ft = get_all_features("aus")
print(len(aus_ft))

# gaze features
gaze_ft = get_all_features("gaze")
print(len(gaze_ft))

# pose features
pose_ft = get_all_features("pose")
print(len(pose_ft))

print(len(X))

first_ft = []
all_features = []
features = []

for idx, id in enumerate(X):
    print(id)
    print(idx)
    aus_gaze = [y for x in [aus_ft[idx], gaze_ft[idx]] for y in x]
    first_ft.append(aus_gaze)
    prev_formant = [y for x in [first_ft[idx], pose_ft[idx]] for y in x]
    features.append([id, prev_formant])

    new_df = pd.DataFrame(features, columns=['Participant_ID', 'all_features_concat'])
    new_df.to_csv("all_video_features2.csv", index=False)
