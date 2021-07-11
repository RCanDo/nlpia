#! python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: TF-IDF
subtitle:
version: 1.0
type: tutorial
keywords: [TF-IDF]
description: |
    Term Frequency - Inverse Document Frequency
remarks:
    - work interactively (in Spyder)
    - install NLPIA, see file 02-Tokenization.py
    -
todo:
sources:
    - title: Natural Language Processing in Action
      chapter: 03 - Math With Vectors (TF-IDF vectors)
      pages: 70-xx
      link: "D:\bib\Python\Natural Language Processing in Action.pdf"
      date: 2019
      authors:
          - fullname: Hobson Lane
          - fullname: Cole Howard
          - fullname: Hannes Max Hapke
      usage: |
          not only copy
    - link:
file:
    usage:
        interactive: True   # if the file is intended to be run interactively e.g. in Spyder
        terminal: False     # if the file is intended to be run in a terminal
    name: 03-TF-IDF.py
    path: D:/Projects/Python/NLPA/
    date: 2020-02-13
    authors:
        - nick: rcando
          fullname: Arkadiusz Kasprzyk
          email:
              - arek@staart.pl
              - akasp666@google.com
"""

#%% p.72
from nltk.tokenize import TreebankWordTokenizer

sentence = \
"""The faster Harry got to the store, the faster Harry, the faster, would get home."""
print(sentence)

#%%
tokenizer = TreebankWordTokenizer()
tokens = tokenizer.tokenize(sentence.lower())
tokens

#%%
from collections import Counter

bag_of_words = Counter(tokens)
bag_of_words

bag_of_words.most_common(4)

#%%
times_harry_appears = bag_of_words['harry']

num_unique_words = len(bag_of_words)

#!!! Term/Token Frequency  within ONE document !!!
tf = times_harry_appears / num_unique_words
round(tf, 4)
2/11

#%%
#%% p.75
# Kite example
from nlpia.data.loaders import kite_text
print(kite_text)

#%%
tokens = tokenizer.tokenize(kite_text.lower())
token_counts = Counter(tokens)
token_counts

#%%
len(token_counts)   # 180
token_counts.most_common(10)

#%%
import nltk
nltk.download('stopwords', quiet=True)
stopwords = nltk.corpus.stopwords.words('english')
len(stopwords)   # 179

#%%
tokens = [w for w in tokens if w not in stopwords]
doc_length = len(tokens)
doc_length  # 222

kite_counts = Counter(tokens)
len(kite_counts)   # 147
kite_counts.most_common(10)

#%% p. 76 Vectorizing
document_vector = []
for k, v in kite_counts.most_common():
    v2 = v / doc_length
    document_vector.append(v2)
    print("{:.4f}".format(v2))


#%%
#%% p. 77 digression on Harry...
docs = [sentence]
docs.append("Harry is hairy and faster then Jill.")
docs.append("Jill is not as hairy as Harry.")

# the same as
from nlpia.data.loaders import harry_docs as docs
docs

#%%
docs_tokens = []
for d in docs:
    docs_tokens += [sorted(tokenizer.tokenize(d.lower()))]
docs_tokens

#%%
all_docs_tokens = sum(docs_tokens, [])                                     #!!!
all_docs_tokens
len(all_docs_tokens)  # 33

#%%  lexicon == unique tokens == a dictionary
lexicon = sorted(set(all_docs_tokens))
lexicon
len(lexicon)  # 18

#%% p. 78
from collections import OrderedDict                                        #!!!
zero_vector = OrderedDict((token, 0) for token in lexicon)
zero_vector

#%%
import copy
docs_vectors = []
for doc in docs:
    zv = copy.copy(zero_vector)
    tokens = tokenizer.tokenize(doc.lower())
    for k, v in Counter(tokens).items():
        zv[k] = v / len(lexicon)              #??? /len(lexicon) ???  why not /len(doc)
    docs_vectors.append(zv)

docs_vectors

## pretty shitty...

dir(docs_vectors[0])
list(docs_vectors[0].values())

#%%
#%% p.82 cosine similarity ~= inner product
"""
$$  \cos\theta = <a, b> / ( |a||b| )  $$
"""
import numpy as np

a = np.array([1, 2, -3])
b = np.array([-2, 3, 1])

a.dot(b)  # 1

# theta = angle_between(a, b)    # how to get it???
# a.dot(b) == np.linalg.norm(a) * np.linalg.norm(b) * np.cos(theta)
# on the other hand

a.dot(b) / np.linalg.norm(a) / np.linalg.norm(b)   # == cos(theta)
# 0.07142857142857144

#%% fundamentally

import math as m

def cosine(a, b):
    res = 0
    for ai, bi in zip(a, b):
        res += ai * bi
    norma = m.sqrt(sum([ai**2 for ai in a]))
    normb = m.sqrt(sum([bi**2 for bi in b]))
    res = res / norma / normb
    return res

cosine(a, b)
# 0.07142857142857144

#%%
#%% p. 83 Zipf's Law

#%% p. 85
from nlpia.book.examples.ch03_zipf import population
population

#%%
import matplotlib.pyplot as plt

plt.plot(population)

def plot_zipf(yy, title="Zipf's law"):
    xx = np.log(np.arange(len(yy)) + 1)

    plt.figure()
    plt.scatter(xx, yy)
    plt.xlabel('log(rank)')
    plt.ylabel('log(values)')
    plt.title(title)

plot_zipf(np.log(population), 'population')


#%% p.85 The Brown Corpus

nltk.download("brown")
from nltk.corpus import brown

brown.words()[:10]
len(brown.words())   # 1 161 192

brown.tagged_words()[:10]   # part-of-speech tagging

dir(brown)

#%%
puncs = set(', . : ; ? ! \' \\ \" ` - -- [ ] ( )'.split())
puncs

puncs.intersection(brown.words())   #!!! Brown Corpus contains punctuations

brown_words = [w.lower() for w in brown.words() if w not in puncs]
brown_counts = Counter(brown_words)
brown_counts.most_common(30)

plot_zipf(np.log(sorted(brown_counts.values(), reverse=True)), 'words')

#%%
#%% Topic modelling

#%% p.87  Kite example continued

from nlpia.data.loaders import kite_text, kite_history
print(kite_history)

#%%
import numpy as np
from typing import List, Tuple, Dict, Set, Union, Optional, NewType

#%%
def terms_freqs(doc: str) -> Dict[str, int]:
    """ Term/Token Frequency within ONE document !!!
    """

    doc = doc.lower()
    tokens = tokenizer.tokenize(doc)
    n = len(tokens)

    counts = Counter(tokens)
    tf = {k: v/n for k, v in counts.items()}   # term freq.

    return tf

#%%
tf_intro = terms_freqs(kite_text)
tf_history = terms_freqs(kite_history)

len(tf_intro)   # 180
len(tf_history) # 172

tf_intro['kite']
tf_history['kite']

tf_intro['and']
tf_history['and']

tf_intro['china']    #! Error
tf_history['china']

#%% tf-idf == Term Frequency - Inverse Document Frequency
import copy

def tfidf(corpus: Dict[str, str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    TF-IDF : Term/Token Frequency - Inverse Document Frequency
    d - document
    c = {d: documents} - corpus, fixed set of documents
    w - word
    l - lexicon = U_d {w: w in d} - all unique words/terms/tokens(?!) in the whole corpus
    N(c) = #c - number of documents in the corpus c
    N(d) = #d - number of words/terms/tokens in the document d
    N(w, d) - number of word w appearances in a document d
    N(w) = #{d: w in d} - number of documents with a word w

    TF(w, d) = N(w, d) / N(d) - word frequency in a document
    DF(w, c) = N(w) / N(c) - frequency of documents with given word wrt whole corpus
    IDF(w, c) = 1 / DF(w, c)

    TFIDF(w, d, c) = TF(w, d) / DF(w, c) == TF(w, d) * IDF(w, c)
    """

    Nc = len(corpus)

    # Term Frequencies {d: {w: TF(w, d), w in d}, d in c}
    TFs = {k: terms_freqs(doc) for k, doc in corpus.items()}   # Dict[Dict[]]

    lexicon = set()
    for tf in TFs.values():
        lexicon |= set(tf.keys())

    # Document Frequency for each word from lexicon
    DFs = {word: 0 for word in lexicon}

    tfidfs = {k: DFs.copy() for k in corpus.keys()}

    # fill  DFs  with document frequencies:
    # = how many documents with a given word
    for word in lexicon:
        DFs[word] = sum(word in doc.keys() for doc in TFs.values())
        #for doc in TFs.values():
        #    if word in doc.keys():
        #        DFs[word] += 1

    # go back to  tfidfs  -- fill with tf-idfs  values
    for doc in tfidfs.keys():
        for word in tfidfs[doc].keys():
            tfidfs[doc][word] = TFs[doc].get(word, 0)
            tfidfs[doc][word] *= (Nc / DFs[word])
            # tf[word] *= np.log(n / DFs[word])   # good for big corpus

    return DFs, tfidfs, TFs


#%%
lex, tfidfs, tfs = tfidf({"intro": kite_text, "history": kite_history})

def tfidf_values(word):
    for doc in ['intro', 'history']:
        print("tfidf in '{}' for '{}' = {:.4}".format(doc, word, tfidfs[doc].get(word, 0.)))
        print("tf in '{}' for '{}' = {:.4}".format(doc, word, tfs[doc].get(word, 0.)))
        print()

tfidf_values('kite')
tfidf_values('and')
tfidf_values('china')

#%%
#%% p.90 Relevance ranking
docs
query = "How long does it take to get to the store?"

def do_corpus(query, docs):
    docs_copy = docs.copy()
    docs_copy.insert(0, query)
    return dict(zip(np.arange(len(docs_copy)), docs_copy))

lex, tfidfs, tfs = tfidf(do_corpus(query, docs))

#%%
def get_answer(query: str, docs: List[str]):
    lex, tfidfs, tfs = tfidf(do_corpus(query, docs))
    mm = np.array(list(list(doc.values()) for doc in tfidfs.values()))
    mmn = mm / np.linalg.norm(mm, axis=1, keepdims=True)
    corrs = mmn[0, :] @ mmn.T    # 0 is for query
    which = corrs[1:].argmax()
    return docs[which], corrs

#%%
ans, corrs = get_answer(query, docs)
ans
corrs

#%%
lex, tfidfs, tfs = tfidf(dict(zip([0, 1, 2], docs)))
lex
tfidfs

#%%
# see  inverted index
#      Whoosh, GitHub.com/Mplsbeb/whoosh
#      additive smoothing

#%% p.93 Tools
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = docs
vectorizer = TfidfVectorizer(min_df=1)
model = vectorizer.fit_transform(corpus)
print(model.todense().round(2))

model     # <3x16 sparse matrix of type '<class 'numpy.float64'>'
    	  # with 23 stored elements in Compressed Sparse Row format>
print(model)
dir(model)

#%%



#%%



#%%



#%%



#%%



