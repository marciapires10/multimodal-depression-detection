import re as re
import string

import contractions
import nltk
import pandas as pd


def sentence_tokenize(file):
    return nltk.tokenize.sent_tokenize(file)


def expand_contractions(file):
    return contractions.fix(file)


def semantics_removal(file):
    text = re.sub('<[^<]+?>', '', file)
    text = re.sub('\[(.*?)\]', '', text)

    return text


def remove_whitespaces(file):
    return nltk.tokenize.WhitespaceTokenizer().tokenize(file)


def lower_casing(tokens):
    return [token.lower() for token in tokens]


def stopwords_removal(tokens):
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.extend(['mmm', 'k', 'xxx'])
    stops = set(stopwords)

    return [token for token in tokens if token not in stops]


def punctuation_removal(tokens):
    text = str.maketrans("", "", string.punctuation)

    return [token.translate(text) for token in tokens]


# read final csv file with all transcripts
df = pd.read_csv('final.csv')

# get the original text from the transcripts
for t in df.Transcript:

    #contractions = expand_contractions(t)
    no_semantics = semantics_removal(t)
    # sentences = sentence_tokenize(no_semantics)
    tokens = remove_whitespaces(no_semantics)
    no_punct = punctuation_removal(tokens)
    lower = lower_casing(no_punct)
    no_stops = stopwords_removal(lower)
    text = ' '.join(no_stops)

    df['Transcript'] = df['Transcript'].replace({t: text})

df.to_csv("final_clean_transcripts.csv", index=False)

#original_text = df.Transcript[1]

# apply text filters
#contractions = expand_contractions(original_text)
#no_semantics = semantics_removal(contractions)
#sentences = sentence_tokenize(no_semantics)
#tokens = remove_whitespaces(no_semantics)
#no_punct = punctuation_removal(tokens)
#lower = lower_casing(no_punct)
#no_stops = stopwords_removal(lower)
#text = ' '.join(no_stops)
#print(text)

#print("Final filtered text:\n", text)

#filtered_text = df.Transcript[0]
