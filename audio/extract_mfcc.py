import librosa as librosa
import numpy as np
import pandas as pd
from librosa import display
from matplotlib import pyplot as plt


# Followed the tutorial: https://www.analyticsvidhya.com/blog/2022/03/implementing-audio-classification-project-using-deep-learning/

# mel frequency cepstral coefficients
def extract_mfcc(file):
    y, sr = librosa.load(file, res_type='kaiser_fast')
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_scaled = np.mean(mfcc.T, axis=0)

    # plt.figure(figsize=(12, 6))
    #
    # plt.subplot(3, 1, 1)
    # display.specshow(mfcc)
    # plt.ylabel('MFCC')
    # plt.colorbar()
    # plt.show()

    return mfccs_scaled


df = pd.read_csv("full_labels.csv")
x = df.Participant_ID
y = df.PHQ8_Binary

extracted_mfcc = []
for idx, id in enumerate(x):
    try:
        path = 'wav_wosilence/{}_wosilence.wav'.format(id)
        print(path)
        extracted_mfcc.append([id, extract_mfcc(path), y[idx]])
    except:
        print("Participant " + id + " doesn't exist.")

mfcc_df = pd.DataFrame(extracted_mfcc, columns=['Participant_ID', 'MFCC', 'PHQ8_Binary'])
mfcc_df.to_csv('mfcc_features.csv')
