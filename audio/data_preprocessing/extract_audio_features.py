import librosa as librosa
import numpy as np
import pandas as pd

from scipy.stats import kurtosis, skew, mode


df = pd.read_csv("../full_labels.csv")
x = df.Participant_ID
y = df.PHQ8_Binary

# For the MFCC, followed the tutorial: https://www.analyticsvidhya.com/blog/2022/03/implementing-audio-classification-project-using-deep-learning/

# extract mel frequency cepstral coefficients from raw audio
def extract_mfcc(file):
    y, sr = librosa.load(file, res_type='kaiser_fast')
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_mean = np.mean(mfcc.T, axis=0)
    mfccs_std = np.std(mfcc.T, axis=0)
    mfccs_skew = skew(mfcc.T, axis=0)
    mfccs_kurtosis = kurtosis(mfcc.T, axis=0)
    mfccs_concat = np.concatenate((mfccs_mean, mfccs_std, mfccs_skew, mfccs_kurtosis), axis=None)

    # plt.figure(figsize=(12, 6))
    #
    # plt.subplot(3, 1, 1)
    # display.specshow(mfcc)
    # plt.ylabel('MFCC')
    # plt.colorbar()
    # plt.show()

    return mfccs_mean, mfccs_std, mfccs_skew, mfccs_kurtosis, mfccs_concat


def save_mfcc_to_csv():
    extracted_mfcc = []
    for idx, id in enumerate(x):
        try:
            path = 'wav_wosilence/{}_final.wav'.format(id)
            print(path)
            extracted_mfcc.append([id, extract_mfcc(path)[0], extract_mfcc(path)[1], extract_mfcc(path)[2],
                                   extract_mfcc(path)[3], extract_mfcc(path)[4], y[idx]])
        except:
            print("Participant " + str(id) + " doesn't exist.")

    mfcc_df = pd.DataFrame(extracted_mfcc, columns=['Participant_ID', 'MFCC_mean', 'MFCC_std', 'MFCC_skew',
                           'MFCC_kurtosis', 'MFCC_concat', 'PHQ8_Binary'])
    mfcc_df.to_csv('mfcc_features.csv')


#save_mfcc_to_csv()

# remove unvoiced segments from covarep features csv
def remove_unvoiced_segments():

    for idx, id in enumerate(x):
        try:
            path = "/media/marciapires/My Passport/audio/covarep/{}_COVAREP.csv".format(id)
            df = pd.read_csv(path, header=None)
            rows = df.iloc[:,1].values

            unvoiced_rows = []
            for i, k in enumerate(rows.tolist()):
                if k == 0:
                    unvoiced_rows.append(i)

            new_df = df.drop(unvoiced_rows)

            new_path = "/media/marciapires/My Passport/audio/{}_cov_voiced.csv".format(id)
            new_df.to_csv(new_path, index=False, header=None)

        except:
            print("Participant " + str(id) + " doesn't exist.")


#remove_unvoiced_segments()


# extract statistical features from covarep/formant csv
def extract_statistical_features(source):

    for idx, id in enumerate(x):
        try:
            if source == "covarep":
                path = "/media/marciapires/My Passport/audio/{}_cov_voiced.csv".format(id)
                new_path = "/media/marciapires/My Passport/audio/{}_cov.csv".format(id)
            else:
                path = "/media/marciapires/My Passport/audio/formant/{}_FORMANT.csv".format(id)
                new_path = "/media/marciapires/My Passport/audio/{}_for.csv".format(id)

            df = pd.read_csv(path, header=None)

            extracted_mean = []
            extracted_std = []
            extracted_skew = []
            extracted_kurtosis = []
            extracted_median = []
            extracted_min = []
            extracted_max = []

            for column in df.columns:
                extracted_mean.append(df[column].mean())
                extracted_std.append(df[column].std())
                extracted_skew.append(df[column].skew())
                extracted_kurtosis.append(df[column].kurtosis())
                extracted_median.append(df[column].median())
                extracted_min.append(df[column].min())
                extracted_max.append(df[column].max())

            new_df = pd.DataFrame([extracted_mean, extracted_std, extracted_skew, extracted_kurtosis, extracted_median,
                                   extracted_min, extracted_max])
            new_df.to_csv(new_path, index=False, header=None)

        except:
            print("Participant " + str(id) + " doesn't exist.")



# run file for covarep csvs
extract_statistical_features("covarep")

# run file for formant csvs
extract_statistical_features("formant")