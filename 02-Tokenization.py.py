#! python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: Toekenization
subtitle:
version: 1.0
type: tutorial
keywords: [tokenization, stemming, lemmatizing,
           n-gram, stop word, term, token, bag of words, sentiment]
description: |
    Basic text preprocessing in examples.
    1. tokenizer
    2. removing stop-words
    3. lemmatizer
    4. stemmer
    5. sentiment
remarks:
    - work interactively (in Spyder)
    - install NLPIA, see sources below
    -
todo:
sources:
    - title: Natural Language Processing in Action
      chapter: 02 - Build your vocabulary (word tokenization)
      pages: 30-69
      link: "D:\bib\Python\Natural Language Processing in Action.pdf"
      date: 2019
      authors:
          - fullname: Hobson Lane
          - fullname: Cole Howard
          - fullname: Hannes Max Hapke
      usage: |
          not only copy
    - title: NLPIA GitHub repository
      link: https://github.com/totalgood/nlpia
    - link: https://github.com/jedijulia/porter-stemmer/blob/master/stemmer.py
file:
    usage:
        interactive: True   # if the file is intended to be run interactively e.g. in Spyder
        terminal: False     # if the file is intended to be run in a terminal
    name: 02-Tokenization.py
    path: D:/Projects/Python/NLPA/
    date: 2020-02-02
    authors:
        - nick: rcando
          fullname: Arkadiusz Kasprzyk
          email:
              - arek@staart.pl
              - akasp666@google.com
"""
#%% p.34

sentence = """Thomas Jefferson begun building Monticello at the age of 26."""
sentence.split()
str.split(sentence)

from collections import Counter
Counter(sentence.split())

#%% p.35

import numpy as np
token_sequence = str.split(sentence)   #!!! tokens seq.   do not have to be different
vocab = sorted(set(token_sequence))    #!!! tokens set    set of different tokens
print(vocab)

num_tokens = len(token_sequence)  #!!! nr of words
vocab_size = len(vocab)           #!!! nr of **different** words

onehot_vectors = np.zeros((num_tokens, vocab_size), int)

for i, word in enumerate(token_sequence):
    onehot_vectors[i, vocab.index(word)] = 1

onehot_vectors

#%% p.36
import pandas as pd
df = pd.DataFrame(onehot_vectors, columns=vocab)
df
df[df==0] = ''
df

#%% p.40 BOW - Bag Of Words (tokens) - set of words/tokens for one document
sentence_bow = {}
for token in sentence.split():
    sentence_bow[token] = 1
# this won't work if tokens repeat in the sentence

sentence_bow
sorted(sentence_bow.items())

#%% p.40
token_freq = [(token, 1) for token in sentence.split()]
# this won't work if tokens repeat in the sentence
token_freq
dict(token_freq)                                      #!
token_freq_s = pd.Series(dict(token_freq))
token_freq_s

df = pd.DataFrame(token_freq_s, columns=['sent'])     #!
df
df.T

#%% p.41
sentences = """Thomas Jefferson begun building Monticello at the age of 26.\n"""
sentences += """Constructions was done mostly by local masons and carpenters.\n"""
sentences += """He moved into the south pavilon in 1770.\n"""
sentences += """Turnig Monticello into a neoclassical masterpiece was Jefferson's obsession."""
print(sentences)

#%%
corpus = {}
for i, sent in enumerate(sentences.split('\n')):
    corpus['sent{}'.format(i)] = dict(Counter(sent.split()))

import pprint as pp
pp.pprint(corpus)

#%%
pd.DataFrame.from_records(corpus)
df = pd.DataFrame.from_records(corpus).fillna(0).astype(int)      #!
df
dft = df.T
dft
dft.iloc[:,:10]

#%% p.42 dot product
v1 = np.array([1, 2, 3])
v1
v2 = np.array([2, 3, 4])
v2
# np is part of pd
pd.np.array([1, 2, 3])

v1.dot(v2)   # it's fast
(v1 * v2)
(v1 * v2).sum()

sum(x1 * x2 for x1, x2 in zip(v1, v2))  # it's slow!

#%% alternatively use matrix product `@`

v1.reshape(-1, 1)
v1.reshape(-1, 1).T  # it's 2-dim
# tha same as
v1.reshape(1, -1)
v1.reshape(-1, 3)

v1.reshape(-1, 1).T @ v2.reshape(-1, 1)  # result is also 2-dim
v1.reshape(1, -1) @ v2.reshape(-1, 1)    # the same

#%% overlap of word counts for two bag-vectors
df.sent0
df.sent0.dot(df.sent1)  # 0  common words
df.sent0.dot(df.sent2)  # 1  common word
df.sent0.dot(df.sent3)  # 1  common word

df.T @ df
df.cov()  # :(
df.T.dot(df)

# find out which word it is
df.sent0 & df.sent3   # pd.Series
(df.sent0 & df.sent3).items()
[(k, v) for (k, v) in (df.sent0 & df.sent3).items()]        #!
dict((k, v) for (k, v) in (df.sent0 & df.sent3).items())        #!
[(k, v) for (k, v) in (df.sent0 & df.sent3).items() if v]
dict((k, v) for (k, v) in (df.sent0 & df.sent3).items() if v)

#%%
#%% token improvment

#%% p.43
import re
sentence

re.split(r' ', sentence)
for w in re.split(r'[-\s.,;!?]+', sentence): print(w)   #!!!

#%% compile regexp
pattern = re.compile(r'[-\s.,;!?]+')
pattern.split(sentence)

#%%
sentence
pattern
tokens = pattern.split(sentence)
tokens = [x for x in tokens if x and x not in '- \t\n.,;!?']  #!!! we still need to rmv sth
tokens

#%% built-in tokenizers
#%% p.46

from nltk.tokenize import RegexpTokenizer   #!!!
tokenizer = RegexpTokenizer(r'\w+|$[0-9.]+|\S+')
tokenizer.tokenize(sentence)

#%% p.47

from nltk.tokenize import TreebankWordTokenizer  #!!!

sentence1 = """Monticello wasn't designated as UNESCO World Heritage Site until 1987."""

tokenizer1 = TreebankWordTokenizer()
tokenizer1.tokenize(sentence1)

#%% p.48 tokenize informal text from social networks - catches emoticons etc.
from nltk.tokenize.casual import casual_tokenize  #!!!

message = """RT @TJMonticello Best day everrrrrr at Monticello.\
Awesommmmmmeeeeee day :*)"""

casual_tokenize(message)
casual_tokenize(message, reduce_len=True, strip_handles=True)

#%%
#%% p.48 n-grams

from nltk.util import ngrams  #!!! generator

list(ngrams(tokens, 2))
list(ngrams(tokens, 3))

#%%
two_grams = list(ngrams(tokens, 2))
[" ".join(tup) for tup in two_grams ]

#%%
#%% p.53 stop words = a, an, the, of, on, in, ...

import nltk
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
len(stop_words)  # 179
stop_words[:20]
[sw for sw in stop_words if len(sw)==1]

nltk.corpus.stopwords.words('polish')  #! No such file or directory: .../stopwords/polish

#%% p.54 sklearn stop words
from sklearn.feature_extraction.text import \
   ENGLISH_STOP_WORDS as sklearn_stop_words

type(sklearn_stop_words)   # frozenset
len(sklearn_stop_words)   # 318
len(sklearn_stop_words.union(stop_words))  # 378
len(sklearn_stop_words.intersection(stop_words))  # 119

#%% p.55 normalization
# case folding
[x.lower() for x in tokens]
# hmmm... problematic

#%% stemming -- searching for the core of the word == getting rid of prefixes, suffixes, etc.

#%% p.58 simple stemmer

re.findall('^(.*ss|.*?)(s)?$', 'houses')
re.findall('^(.*ss|.*)(s)?$', 'houses')      #! NO!!!

re.findall('^(.*ss|.*?)(s)?$', 'housess')
re.findall('^(.*ss|.*)(s)?$', 'housess')     # the same


def stem(phrase):
    tokens = phrase.lower().split()
    tokens2 = [re.findall('^(.*ss|.*?)(s)?$', word)[0][0].strip("'") for word in tokens]
    return " ".join(tokens2)

#%%
stem('houses')
stem("Doctor House's calls")

#%%
from nltk.stem.porter import PorterStemmer   #!!!
stemmer = PorterStemmer()

phrase = "dish washer's washing dishes"
[stemmer.stem(w) for w in phrase.split()]
' '.join([stemmer.stem(w).strip("'") for w in phrase.split()])

# see: https://github.com/jedijulia/porter-stemmer/blob/master/stemmer.py

#%% lemmatization -- searching for the core meaning / root of the word
#!!! use before stemmer !!!

#%% p.61
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer   #!!!
lemmatizer = WordNetLemmatizer()

#%%
lemmatizer.lemmatize('better')             # better,  pos='n' (noun) is default
lemmatizer.lemmatize('better', pos='a')    # good
lemmatizer.lemmatize('good')               # good
lemmatizer.lemmatize('good', pos='a')      # good
lemmatizer.lemmatize('goods', pos='n')     # good
lemmatizer.lemmatize('goods', pos='a')     # goods
lemmatizer.lemmatize('goodness', pos='a')  # goodness
lemmatizer.lemmatize('goodness', pos='n')  # goodness

stemmer.stem('goodness')                # good

lemmatizer.lemmatize('best', pos='n')      # best
lemmatizer.lemmatize('best', pos='a')      # best

#%%
lemmatizer.lemmatize('do', pos='n')      # do
lemmatizer.lemmatize('do', pos='a')      # do
lemmatizer.lemmatize('do', pos='v')      # do

lemmatizer.lemmatize('did', pos='n')      # did
lemmatizer.lemmatize('did', pos='a')      # did
lemmatizer.lemmatize('did', pos='v')      # do

lemmatizer.lemmatize('done', pos='n')      # done
lemmatizer.lemmatize('done', pos='a')      # done
lemmatizer.lemmatize('done', pos='v')      # do

#%%
lemmatizer.lemmatize('has')             # ha  #???
lemmatizer.lemmatize('has', pos='v')    # have
lemmatizer.lemmatize('has', pos='a')    # has
lemmatizer.lemmatize("hasn't")          # hasn't
stemmer.stem("hasn't")
lemmatizer.lemmatize("wasn't")          # wasn't
stemmer.stem("don't")

#%%
lemmatizer.lemmatize("dont't")          # don't
stemmer.stem("don't")
lemmatizer.lemmatize("fall back")          # fall back


#%%
#%% sentiment analysis

#%% p.64 VADER -- rule based sentiment analyzer
"""
nltk.sentiment.vader
# but here we use
pip install vaderSentiment
"""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  #!!!
sa = SentimentIntensityAnalyzer()

#%%
sa.lexicon
type(sa.lexicon)   # dict
#!!! neither lemmatized nor stemmed (bad); no stop-words (good)

len(sa.lexicon)    # 7503

# only tokens with space i.e. few words
[(tok, score) for tok, score in sa.lexicon.items() if " " in tok]

#%%
text1 = """Python is very readable and it's great for NLP."""
text2 = """Python is not a bad choice for most applications."""

sa.polarity_scores(text=text1)
sa.polarity_scores(text=text2)

#%%
corpus = ["Absolutely perfect! Love it! :-) :-) :-)",
          "Horrible! Completely useless. :(",
          "It was OK. Some good and some bad things."]

#%%
for doc in corpus:
    scores = sa.polarity_scores(doc)
    print("{:+}: {}".format(scores['compound'], doc))


#%%
#%% p.65 Naive Bayes

# using nlpia package - one must create nlpiaenv environment first...

from nlpia.data.loaders import get_data
movies = get_data('hutto_movies')

#%%
type(movies)     # pd.DataFrame
movies.shape     # 10605, 2
movies.columns

movies.head().round(2)
movies.describe().round(2)    # scores between -4 and 4

movies.sentiment.plot.kde()
movies.sentiment.plot.hist(bins=30)

#%%
import pandas as pd
# pd.set_option('display.width', 300)
# pd.options.display.width = 300
pd.set_option('display.max_columns', 300)
pd.set_option('display.max_rows', 300)

#%%
from nltk.tokenize import casual_tokenize   #!!!
from collections import Counter

# BOW - collection of all words (tokens!) from all docs in corpus
bags_of_words = []
for txt in movies.text:
    bags_of_words.append(Counter(casual_tokenize(txt)))

len(bags_of_words)  # 10605
bags_of_words[:5]

#%% to data frame
df_bows = pd.DataFrame.from_records(bags_of_words)
df_bows = df_bows.fillna(0).astype(int)
df_bows.shape     # 10605, 20756
df_bows.head()

#%%
"""
Notice that we didn't make any tokens improvements: only `casual_tokenizer`,
_stop words_ not removed; not transformed to lower case;
neither _stemmers_ nor _lemmatizers_ applied.
"""

#%%
from sklearn.naive_bayes import MultinomialNB
model01 = MultinomialNB()
type(model01)     # sklearn.naive_bayes.MultinomialNB
dir(model01)
model01.fit(df_bows, movies.sentiment > 0)     # .fit()  works in place

#%% comparing fit with true
model01.predict_proba(df_bows)
model01.classes_  # [False, True]
model01.predict_proba(df_bows)[:5, 1]   # score for True i.e. positive sentiment
model01.predict_proba(df_bows)[:5, 1] * 8 - 4  # scaling to (-4, 4)
movies.iloc[:5, :]   # more or less...
movies['text'][1]

model01.predict_proba(df_bows)[-5:, 1]   # score for True i.e. positive sentiment
model01.predict_proba(df_bows)[-5:, 1] * 8 - 4  # scaling to (-4, 4)
movies.iloc[-5:, :]

#%% other options of model.  !!! check it !!!
model01.coef_
model01.coef_.shape   # (1, 20756)
model01.intercept_
model01.get_params()
model01.class_prior
model01.classes_
model01.n_features_    # 20756

#%%
movies['predicted_sentiment'] = model01.predict_proba(df_bows)[:, 1] * 8 - 4  # scaling to (-4, 4)
movies['error'] = (movies.predicted_sentiment - movies.sentiment).abs()
movies.error.mean().round(2)   # 1.87

#%%
movies['sentiment_is_positive'] = (movies.sentiment > 0).astype(int)
movies['predicted_is_positive'] = (movies.predicted_sentiment > 0).astype(int)
movies['prediction_wrong'] = (movies.sentiment_is_positive - movies.predicted_is_positive).astype(int).abs()
movies.prediction_wrong.sum()   # 695

movies.prediction_wrong.abs().sum()/len(movies)        # 0.0655   error rate
1 - movies.prediction_wrong.abs().sum()/len(movies)    # 0.93446  correct rate

movies.columns
movies.iloc[:99, [0, 2, 4, 5, 6]]


#%% check the model on the other data set
# this time  product reviews
product = get_data('hutto_products')
product   # 3546, 3
bags_of_words_2 = []

for txt in product.text:
    bags_of_words_2.append(Counter(casual_tokenize(txt)))

#%%
df_bows_2 = pd.DataFrame.from_records(bags_of_words_2)
df_bows_2 = df_bows_2.fillna(0).astype(int)

#%%
df_all_bows = df_bows.append(df_bows_2)
df_all_bows.columns    # 23302

#%%
df_bows_22 = df_all_bows.iloc[len(movies):][df_bows.columns]
df_bows_22.shape     # 3546, 20756

#%% check it        !!!
df_bows_22.info()
df_bows_22.index         #
df_bows_22.index[-1]     #
len(df_bows_22.index)    # 3546
df_bows_22.shape[0]

# !!!
df_bows_22.count()         # non NaNs by columns; reverse is:
df_bows_22.isnull().sum()  # many NaNs;  or
len(df_bows_22.index) - df_bows_22.count()

all(df_bows_22.columns == df_bows.columns)    # True

#%% so we need
df_bows_22 = df_bows_22.fillna(0).astype(int)

#%%
product['sentiment_is_positive'] = (product.sentiment > 0).astype(int)
product['predicted_sentiment'] = model01.predict_proba(df_bows_22)[:, 1] * 8 - 4
product['predicted_is_positive'] = (product.predicted_sentiment > 0).astype(int)

product['prediction_wrong'] = (product.sentiment_is_positive - product.predicted_is_positive).astype(int).abs()

product['prediction_wrong'].sum()
product['prediction_wrong'].sum()/3546   # 0.44275

# no good...

#%%

