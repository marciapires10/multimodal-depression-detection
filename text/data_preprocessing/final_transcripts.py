import pandas as pd

full_label = pd.read_csv('full_labels.csv')
all_transcripts = pd.read_csv('transcripts/all_transcripts.csv')

final = pd.merge(full_label, all_transcripts, on='Participant_ID', how='inner')

final = final.sort_values(['Participant_ID'], ascending=True)
final = final.reset_index(drop=True)
final.drop("Unnamed: 0", axis=1, inplace=True)
final.to_csv('final.csv', index=False)

