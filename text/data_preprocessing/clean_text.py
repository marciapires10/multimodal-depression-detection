import re as re
import string
from collections import Counter

import contractions

import nltk
import pandas as pd
from nltk import PorterStemmer
from nltk.stem import WordNetLemmatizer


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
    stopwords.extend(['mmm', 'mm', 'hmm', 'mhm', 'k', 'xxx', 'um', 'uh'])
    stops = set(stopwords)

    return [token for token in tokens if token not in stops]


def punctuation_removal(tokens):
    text = str.maketrans("", "", string.punctuation)

    return [token.translate(text) for token in tokens]


def lemmatization(tokens):
    lemmatizer = WordNetLemmatizer()

    return [lemmatizer.lemmatize(token) for token in tokens]


def stemming(tokens):
    ps = PorterStemmer()

    return [ps.stem(token) for token in tokens]


# read final csv file with all transcripts
df = pd.read_csv('final.csv')

# get the original text from the transcripts
for t in df.Transcript:
    _contractions = expand_contractions(t)
    no_semantics = semantics_removal(_contractions)
    # sentences = sentence_tokenize(no_semantics)
    tokens = remove_whitespaces(no_semantics)
    no_punct = punctuation_removal(tokens)
    lower = lower_casing(no_punct)
    no_stops = stopwords_removal(lower)
    lemm = lemmatization(no_stops)
    #stem = stemming(no_stops)
    text = ' '.join(lemm)

    df['Transcript'] = df['Transcript'].replace({t: text})

df.to_csv("final_clean_transcripts3.csv", index=False)

new_df = pd.read_csv('final_clean_transcripts3.csv')

X = new_df.Transcript

cnt = Counter()
for text in X.values:
    for word in text.split():
        cnt[word] += 1
FREQWORDS = set([w for (w, wc) in cnt.most_common(20)])


def remove_freqwords(text):
    return " ".join([word for word in str(text).split() if word not in FREQWORDS])


X = X.apply(lambda text: remove_freqwords(text))

n_rare_words = 100
cnt = Counter()
for text in X.values:
    for word in text.split():
        cnt[word] += 1
RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-n_rare_words - 1:-1]])


def remove_rarewords(text):
    return " ".join([word for word in str(text).split() if word not in RAREWORDS])


X = X.apply(lambda text: remove_rarewords(text))


new_df['Transcript'] = X
new_df.to_csv("final_clean_transcripts5.csv", index=False)

# original_text = df.Transcript[1]
# print(original_text)
#
# # apply text filters
# contractions = expand_contractions(original_text)
# print(contractions)
# no_semantics = semantics_removal(contractions)
# sentences = sentence_tokenize(no_semantics)
# tokens = remove_whitespaces(no_semantics)
# no_punct = punctuation_removal(tokens)
# lower = lower_casing(no_punct)
# no_stops = stopwords_removal(lower)
# text = ' '.join(no_stops)
# print(text)
#
# print("Final filtered text:\n", text)
