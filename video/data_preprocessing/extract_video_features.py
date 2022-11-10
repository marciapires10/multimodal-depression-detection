import pandas as pd

df = pd.read_csv("../full_labels.csv")
x = df.Participant_ID
y = df.PHQ8_Binary


# extract statistical features from covarep/formant csv
def extract_statistical_features(source):

    for idx, id in enumerate(x):
        try:
            if source == "aus":
                path = "/media/marciapires/My Passport/video/aus/{}_CLNF_AUs.txt".format(id)
                new_path = "/media/marciapires/My Passport/video/{}_aus.csv".format(id)
            elif source == "gaze":
                path = "/media/marciapires/My Passport/video/gaze/{}_CLNF_gaze.txt".format(id)
                new_path = "/media/marciapires/My Passport/video/{}_gaze.csv".format(id)
            else:
                path = "/media/marciapires/My Passport/video/pose/{}_CLNF_pose.txt".format(id)
                new_path = "/media/marciapires/My Passport/video/{}_pose.csv".format(id)


            df = pd.read_csv(path)
            extracted_mean = []
            extracted_std = []
            extracted_skew = []
            extracted_kurtosis = []
            extracted_median = []
            extracted_min = []
            extracted_max = []
            extracted_variance = []

            for column in df.columns:
                #print(df.dtypes)
                df[column] = pd.to_numeric(df[column], errors='coerce')
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


# run file for AUs csvs
extract_statistical_features("aus")

# run file for gaze csvs
extract_statistical_features("gaze")

# run file for pose csvs
extract_statistical_features("pose")


