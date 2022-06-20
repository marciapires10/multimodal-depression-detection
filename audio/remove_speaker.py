import os
import re

from pydub import AudioSegment
import pandas as pd

#transcripts_df = pd.read_csv("../text/data_preprocessing/transcripts/")
transcripts_path = "../text/data_preprocessing/transcripts/"
transcripts_list = os.listdir(transcripts_path)

transcripts_list = sorted(transcripts_list)


for t in transcripts_list:
    transcript_path = os.path.join(transcripts_path, t)
    if not transcript_path == "../text/data_preprocessing/transcripts/all_transcripts.csv":
        transcript = pd.read_csv(transcript_path, sep='\t')
        participant_id = t[0:3]
        # speaker = transcript[transcript.columns[2]]

        combination = AudioSegment.empty()
        for idx in range(0, len(transcript)):
            speaker = transcript.iloc[idx]["speaker"]

            if speaker == "Participant":
                #orig_audio = AudioSegment.from_wav("wav/" + str(participant_id) + "_AUDIO.wav")
                orig_audio = AudioSegment.from_wav("/media/marciapires/My Passport/audio/wav/" + str(participant_id) + "_AUDIO.wav")

                t1 = int(float(transcript.iloc[idx]["start_time"]) * 1000)
                t2 = int(float(transcript.iloc[idx]["stop_time"]) * 1000)

                new_audio = orig_audio[t1:t2]
                combination = combination + new_audio

                combination.export("wav_wosilence/" + str(participant_id) + "_final.wav", format="wav")








