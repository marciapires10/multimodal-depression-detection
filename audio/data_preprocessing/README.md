## Contents
- <b>remove_speaker.py</b>: removes the speaker from the raw audio 
- <b>extract_audio_features.py</b>: creates a csv file (<b>mfcc_features.csv</b>) containing the extracted mfcc 
features from the audio without the speaker; the unvoiced segments from the covarep features are removed resulting in another csv file
  (<b>participantID_cov_voiced.csv</b>); finally, some statistical features are extracted from the covarep and formant
features and two more csv files are generated (<b>participantID_cov.csv</b> and <b>participantID_for.csv</b>)
- <b>feature_selection.py</b>: get the low variance features from the covarep and remove then; then get the f1 regression
features and remove the features with lower values
- <b>concat_audio_features.py</b>: merge all the audio features: mfcc + covarep (after feature selection) + formant
