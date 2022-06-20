import os
from _csv import writer

import pandas as pd


def append_transcripts(filename, values):
    with open(filename, mode='a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(values)


# current directory
#curr_directory = os.getcwd()
#files = os.listdir(curr_directory)

# get transcripts files at 'transcripts' folder
transcripts_path = 'transcripts'
move_to = os.chdir(transcripts_path)
files = os.listdir(move_to)

# sort files
files = sorted(files)

for f in files:
    if f.endswith('.csv'):
        participant_id = f[0:3]
        all_transcripts = ''

        df = pd.read_csv(f, sep='\t')
        drop_columns = ['start_time', 'stop_time']
        df_dropped_columns = df.drop(drop_columns, axis=1)
        df_dropped_ellie = df_dropped_columns.drop(df_dropped_columns.index[df_dropped_columns['speaker'] == 'Ellie'])
        clean_transcript = df_dropped_ellie.reset_index()

        for i, r in clean_transcript.iterrows():
            if i == 0:
                all_transcripts += str(r['value'])
            else:
                all_transcripts += ' ' + str(r['value'])

        append_transcripts('all_transcripts.csv', [participant_id, all_transcripts])

file = pd.read_csv('all_transcripts.csv', header=None)
headerList = ['Participant_ID', 'Transcript']
file.to_csv('all_transcripts.csv', header=headerList, index=False)

