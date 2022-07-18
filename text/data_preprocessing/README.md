## Contents
- <b>clean_transcripts.py</b>: creates a csv file (<b>all_transcripts.csv</b>) containing all the participants id and respective transcript.
- <b>concat_labels.py</b>: creates a csv file (<b>full_labels.csv</b>) containing only the relevant labels (Participant ID, PHQ8_Binary, PHQ8_Score, Gender).
- <b>final_transcripts.py</b>: merge of all_transcripts.csv and full_labels.csv to create a csv file (<b>final.csv</b>) containing the full information needed.
- <b>clean_text.py</b>: set of functions to clean the text from all the transcripts (tokenization, expansion of contractions, semantics removal, stopwords removal, etc.). Returns a file (<b>final.csv</b>) with the resulting texts.
