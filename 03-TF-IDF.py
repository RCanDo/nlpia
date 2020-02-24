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
from collections import OrderedDict
zero_vector = OrderedDict((token, 0) for token in lexicon)
zero_vector

#%%
import copy
docs_vectors = []
for d in docs:
    zv = copy.copy(zero_vector)
    tokens = tokenizer.tokenize(d.lower())
    for k, v in Counter(tokens).items():
        zv[k] = v / len(lexicon)
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

#%%
puncs = set(', . : ; ? ! \' \\ \" ` - -- [ ] ( )'.split())
puncs

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

tf_intro['china']
tf_history['china']

#%% tf-idf == Term Frequency - Inverse Document Frequency
import copy

def tfidf(corpus: Dict[str, str]) -> Tuple[Dict[str, float], Dict[str, float]]:

    n = len(corpus)
    tfs = {k: terms_freqs(doc) for k, doc in corpus.items()}

    lexicon = set()
    for tf in tfs.values():
        lexicon |= set(tf.keys())

    lex_doc_freqs = {word: 0 for word in lexicon}

    tfidfs = {k: lex_doc_freqs.copy() for k in corpus.keys()}

    # fill  lex_doc_freqs  with document frequencies
    for word in lexicon:
        for doc in tfs.values():
            if word in doc.keys():
                lex_doc_freqs[word] += 1

    # go back to  tfidfs  -- fill wit  tf-idfs  values
    for doc in tfidfs.keys():
        for word in tfidfs[doc].keys():
            tfidfs[doc][word] = tfs[doc].get(word, 0)
            tfidfs[doc][word] *= (n / lex_doc_freqs[word])
            # tf[word] *= np.log(n / lex_doc_freqs[word])   # good for big corpus

    return lex_doc_freqs, tfidfs, tfs


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
    corrs = mmn[0, :] @ mmn.T
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

#%%



#%%



#%%



#%%



#%%



