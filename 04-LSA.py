-*- coding: utf-8 -*-
"""
Created on Fri Jun 25 09:55:59 2021

@author: staar
"""

#! python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: Latent Semantic Analysis
subtitle:
version: 1.0
type: tutorial
keywords: [LSA, semantic, topic, TF-IDF]
description: |
    Latent Semantic Analysis
remarks:
    - work interactively (in Spyder)
    - install NLPIA, see file 02-Tokenization.py
    -
todo:
sources:
    - title: Natural Language Processing in Action
      chapter: 04 - Finding meaning in word counts (semantic analysis)
      pages: 97-xx
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
    name: 04-LSA.py
    path: D:/Projects/Python/NLPA/
    date: 2021-06-25
    authors:
        - nick: rcando
          fullname: Arkadiusz Kasprzyk
          email:
              - arek@staart.pl
              - akasp666@google.com
"""
#%%
import numpy as np
import pandas as pd
from collections import namedtuple

import matplotlib.pyplot as plt
import seaborn as sn

#%%
pd.options.display.width = 120

#%%
""" (1)
T = M * F,   (t x d) = (t x n) * (n x d),  d = len(corpus), n=len(vocab), t=len(topics)
topic-document scores
    topic-words weights (aquired from ...? - to be shown later)
        words-document weights - usually TF-IDF; i.e.
        F = [f_1 ... f_d],  f_i - TF-IDF vector for i-th doc
            where vocab is taken from the whole corpus
"""
#%% p.101 thought experiment
topic = {}
TFIDF = namedtuple(typename='TFIDF', field_names='cat dog apple lion NYC love')
TFIDF._fields

# random example of TF-IDF list (for some imaginary doc)
tfidf = TFIDF(*np.random.rand(6).round(3))
tfidf
pd.Series(tfidf)  # names lost :(
pd.Series(tfidf, index=tfidf._fields)   # this should be automatic !!!

# topic weights for each word (arbitrary example)
M = pd.DataFrame([[.3, .3, 0, 0, -.2, .2],
                  [.1, .1, -.1, .5, .1, -.1],
                  [0, -.1, .2, -.1, .5, .1]],
                 index = ['petness', 'animalness', 'cityness'],
                 columns = 'cat dog apple lion NYC love'.split())
M

#!!! topic score for a document
M @ tfidf  # luckily it works for named tuples

#%%
#%% How to acquire weights for M?

#%%
#%% LDA classifier (Linear Disccriminant Analysis)

from nlpia.data.loaders import get_data
sms = get_data('sms-spam')
type(sms)   # pandas.core.frame.DataFrame
sms         # [4837 rows x 2 columns]
sms.dtypes  # spam     int64;  text    object

sms.spam.sum()   # 638

#%%
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize

tfidf = TfidfVectorizer(tokenizer = casual_tokenize)
dir(tfidf)
tfidf.vocabulary  # None
tfidf.fit(raw_documents = sms.text)
tfidf.vocabulary  # None
tfidf.vocabulary_  # whole vocab {word: index}
len(tfidf.vocabulary_)  # 9232
 plt.plot(sorted(tfidf.vocabulary_.values()))
 # so why this dict is so mixed ? why it's a dict not just list ?

sms_tfidf = tfidf.transform(raw_documents = sms.text)
sms_tfidf
 # <4837x9232 sparse matrix of type '<class 'numpy.float64'>'
 #	with 82353 stored elements in Compressed Sparse Row format>
dir(sms_tfidf)
sms_tfidf.shape   # (4837, 9232)

"""
Thus we obtain matrix with
4837 rows for each doc and
9232 cols for each word/term/token from the lexicon for this corpus (of 4837 docs).
This is transposed wrt to our convention for TFIDF ~= [terms x docs]
"""

sms_tfidf_arr = sms_tfidf.toarray()
sms_tfidf_arr
plt.plot(sms_tfidf_arr[0])       # first row ~ doc
plt.plot(sms_tfidf_arr[:, 0])    # first col ~ token

#%% heatmap - takes a lot of time !!!
fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
res = sn.heatmap(sms_tfidf_arr) #, cmap='YlGnBu')
#! Wow! It's hard to see any points different from 0 - very sparse matrix


#%% LDA - Linear Discriminant Analysis
mask = sms.spam.astype(bool).values
mask
centroid_spam = sms_tfidf[mask].mean(axis=0)       # matrix([[0.06377591, 0.0041675 , 0.00056204, ..., 0., 0.,0.]])
centroid_spam = sms_tfidf_arr[mask].mean(axis=0)   # array(  [0.06377591, 0.0041675 , 0.00056204, ..., 0., 0.,0.] )
centroid_ham  = sms_tfidf_arr[~mask].mean(axis=0)  # array

# model
direction = centroid_spam - centroid_ham
# score
spaminess_score = sms_tfidf_arr.dot(direction)     #!!!

spaminess_score[mask].mean()    #  0.0362
spaminess_score[~mask].mean()   # -0.00558

#%%
cols = ['red' if m else 'black' for m in mask]
plt.scatter(x=range(len(spaminess_score)), y=spaminess_score, s=.5, c=cols)
#TODO better plot...

#%%
from sklearn.preprocessing import MinMaxScaler

sms['lda_score'] = MinMaxScaler().fit_transform(spaminess_score.reshape(-1, 1))
sms['lda_predict'] = (sms['lda_score'] > .5).astype(int)
sms.head(20)

errors = sum(abs(sms['spam'] - sms['lda_predict']))
errors   # 109
err_rate = errors/len(mask)
err_rate        # 0.0225
1 - err_rate    # 0.977

confusion = pd.crosstab(sms['spam'], sms['lda_predict'])   #!!!
confusion

# or
from pugnlp.stats import Confusion     #???
Confusion(sms[['spam', 'lda_predict']])

#%%
#import seaborn as sn
fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
res = sn.heatmap(confusion, annot=True, fmt='.0f', cbar=False, cmap='YlGnBu')

#%%
#%% topic-word matrix for LSA on 16 short sentences about cats, dogs and NYC
from nlpia.book.examples.ch04_catdog_lsa_3x6x16 import word_topic_vectors
word_topic_vectors
word_topic_vectors.T.round(1)

# compare with weights matrix W above
# still we don't know where these weights are taken from

#%%
#%%
""" (2)
SVD - SINGULAR VALUE DECOMPOSITION
----------------------------------
https://en.wikipedia.org/wiki/Singular_value_decomposition

Transposition is `t`, `*` is conjugate transpose; both are the same for real matrices;
`'` does NOT mean transposition here - it's only a sign;

W = U S V*   (m x n) = m2 * (m x n) * n2
U = [l_1, ..., l_m] - left singular vectors = left eigenvectors of WW*
V = [r_1, ..., r_n] - righ singular vectors = right eigenvectors of W*W
S = diag(s_1, ..., s_r; m, n), r <= min(m, n) - singular values of W
    - square roots of the non-zero eigenvalues of both WW* and W*W
      s_1 >= s_2 >= ... >= s_r
This implies that
 W r_i = s_i l_i
W* l_i = s_i r_i

Then we may reduce dim taking only p larges singular values for some p <= min(m, n):
W' = U' S' V'*   (m x p) = (m x p) * p2 * (p x n)
i.e.
S' = diag(s_1, ..., s_p)  - square!

In the LSA context
------------------
W - (m x n) word-document matrix; TF-IDF or BOW;
    every possible preprocessing should be applied - tokenise, stem, lemmatise, remove stop-words !
    to avoid colinearity search for:
        pairs/triplets/.. of words which always and only ever appear together (in the whole corpus)
    like, e.g. 'ad hoc', 'status quo', 'merry christmass', ...
    (why exactly it makes problem ??? )

U - m2  word-topic matrix - as many topics as words ! (before reducing dim)
S - (m x n)  variation of topics, from largest to smallest;
Vt == V* - n2  topic-document matrix - as many topics as documents ! (before reducing dim)

Hence 'topic' is a bit vague abstract concept above as their number differ between U and Vt matrices.
However when we reduce dim by clipping to only p largest singular values, s_1, ..., s_p,
then we end up with well determined number of 'topics', namely p:
U'  - (m x p)  word-topic matrix
S'  - p2       topic  variances
Vt' - (p x n)  topic-document matrix

Notice that comparing with (1) we have analogy:
Vt' ~= T  (p x n)  topic-document
Ut' ~= M  (p x m)  topic-word  weights

"""

#%%
from nlpia.book.examples.ch04_catdog_lsa_sorted import lsa_models, prettify_tdm

# LSA on cats_and_dogs corpus
bow_svd, tfidf_svd = lsa_models()
bow_svd

prettify_tdm(**bow_svd)

bow_svd['tdm']          # (m x n)  word-doc
bow_svd['u'].round(2)   # m2       word-topic
bow_svd['s'].round(2)   # r = min(m, n)   topic variations; just vector - may be transform to (m x n) as below (*)
bow_svd['vt'].round(2)  # n2       topic-doc

#%%
import numpy as np
U, s, Vt = np.linalg.svd(bow_svd['tdm'])
U.round(2)  #
s.round(2)
Vt.round(2)

#%% (*)
S = np.zeros((len(U), len(Vt)))
np.fill_diagonal(S, s)
S.round(2)

#%%
W = U @ S @ Vt
W.round(1)
bow_svd['tdm']           #!!! OK !!!

#%% Term-document matrix reconstruction error, p.122
errs = np.array([])
Sprim = S.copy()
for d in range(len(s), 0, -1):  # print(d)
    print(Sprim.round(2))
    Wprim = U @ Sprim @ Vt
    err = np.sqrt( sum((bow_svd['tdm'] - Wprim).values.flatten() ** 2) )
    print(err)
    errs += [err]
    if d > 0:
        Sprim[d-1, d-1] = 0
print([round(x, 2) for x in errs])

#%%b
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize

tfidf = TfidfVectorizer(tokenizer = casual_tokenize)

tfidf.fit(raw_documents = bow_svd['docs'])

#%%

#%%

#%%


#%%