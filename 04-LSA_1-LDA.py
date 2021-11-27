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
keywords: [LSA, semantic, topic, TF-IDF, LDA]
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
      pages: 97-115
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
from rcando.ak.builtin import * #flatten, paste
from rcando.ak.nppd import * #data_frame
import os, sys, json

ROOT = json.load(open('root.json'))
WD = os.path.join(ROOT['Projects'], "AIML/NLPA/")   #!!! adjust
os.chdir(WD)

print(os.getcwd())

#%%
import numpy as np
import pandas as pd
from collections import namedtuple

"""
link: https://stackoverflow.com/questions/11707586/how-do-i-expand-the-output-display-to-see-more-columns-of-a-pandas-dataframe
"""
#pd.options.display.width = 0  # autodetects the size of your terminal window - does it work???
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
# pd.options.display.max_rows = 500         # the same
pd.set_option('display.max_seq_items', None)

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.precision', 3)

# %% other df options
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', None)
#pd.options.display.max_colwidth = 500
# the same

#%%
import matplotlib.pyplot as plt
plt.style.use('dark_background')
# see `plt.style.available` for list of available styles

import seaborn as sn

#%%
pd.options.display.width = 120

#%%
""" (1)
T = M * F,   (t x d) = (t x v) * (v x d),  d = len(corpus), v=len(vocab), t=len(topics)
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
#%% LDA classifier (Linear Discriminant Analysis)

from nlpia.data.loaders import get_data
sms = get_data('sms-spam')
type(sms)   # pandas.core.frame.DataFrame
sms.shape         # [4837 rows x 2 columns]
sms.head()
sms.dtypes  # spam     int64;  text    object

sms.spam.sum()   # 638
638/4837         # 0.132

#%%
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize

tfidf = TfidfVectorizer(tokenizer = casual_tokenize)
dir(tfidf)
tfidf.vocabulary   # None
tfidf.fit(raw_documents = sms.text)
tfidf.vocabulary   # None
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

#%%
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
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

colors = ["black", "cyan", "red", "white"]
nodes = [0.0, 0.01, 0.1, 1.0]   # 'centers' of colors
cmap = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

fig = plt.figure(figsize=[17, 11], tight_layout=True)
#plt.clf()
ax = fig.add_subplot(111)

mappable = ax.pcolormesh(sms_tfidf_arr, cmap=cmap, rasterized=True, vmin=0, vmax=1)
fig.colorbar(mappable, ax=ax)

#%%
#%% LDA - Linear Discriminant Analysis
mask = sms.spam.astype(bool).values
mask
centroid_spam = sms_tfidf[mask].mean(axis=0)       # matrix([[0.06377591, 0.0041675 , 0.00056204, ..., 0., 0.,0.]])
centroid_spam = sms_tfidf_arr[mask].mean(axis=0)   # array(  [0.06377591, 0.0041675 , 0.00056204, ..., 0., 0.,0.] )
centroid_ham  = sms_tfidf_arr[~mask].mean(axis=0)  # array

# "model"
direction = centroid_spam - centroid_ham
plt.plot(direction)

#%% score
spaminess_score = sms_tfidf_arr.dot(direction)     #!!!

spaminess_score[mask].mean()    #  0.0362
spaminess_score[~mask].mean()   # -0.00558

""" !!!
This is substantial oversimplification -- we estimated only direction
and no constant!
It may only work if `direction` is orthogonal to the overall mean of all data.
Luckily it is the case here !!! see (*) below
"""

#%%
cols = ['red' if m else 'yellow' for m in mask]
plt.scatter(x=range(len(spaminess_score)), y=spaminess_score, s=.5, c=cols)
#TODO better plot...

#%%
#%%
from sklearn.preprocessing import MinMaxScaler

sms['lda_score'] = MinMaxScaler().fit_transform(spaminess_score.reshape(-1, 1))
sms['lda_predict'] = (sms['lda_score'] > .5).astype(int)
sms.head(20)

errors = sum( sms['spam'] != sms['lda_predict'] )
errors   # 109
err_rate = errors/len(mask)
err_rate        # 0.0225
1 - err_rate    # 0.977

confusion = pd.crosstab(sms['spam'], sms['lda_predict'])                       #!!!
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
#%% do the same on centralised data
all_mean = sms_tfidf_arr.mean(axis=0)
plt.plot(all_mean)

all_mean.dot(direction)   # -7.090445918691126e-05
"""!!! almost othogonal -- is this coincidence or STH DEEPER with TF-IDF ???     (*)
Notice that TF-IDF works completely agnostic of any labels (here 'spam').
It means that the example above is somehow designed (more or less).

Normally, there is no guarantee that overall mean is orthogonal to the
`direction` between classes -- it would be a miracle!

The proper LDA estimates this overall mean
via estimating surface of discrimination
which has its `direction` parameter (say  A ) together with `constant`
(say B in  Ax + B = 0  linear equation of this surface).

For this data we got  B ~= 0.
"""
#%% extracting mean from all observations, here: rows !!!
sms_tfidf_arr_0 = sms_tfidf_arr - np.tile(all_mean, (sms_tfidf.shape[0], 1))
# sms_tfidf_arr_0.shape   # (4837, 9232)
spaminess_score_0 = sms_tfidf_arr.dot(direction)     #!!!

spaminess_score_0[mask].mean()    #  0.0362
spaminess_score_0[~mask].mean()   # -0.00558
# the same values as for non-cenered data -- what is obvious in view of  (*)

fig = plt.figure()
ax = fig.add_subplot(111)
cols = ['red' if m else 'yellow' for m in mask]
ax.scatter(x=range(len(spaminess_score_0)), y=spaminess_score_0, s=.5, c=cols)

#%%
#%% proper LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
LDA = LinearDiscriminantAnalysis()
lda = LDA.fit(sms_tfidf_arr, mask)
sms['lda'] = lda.predict(sms_tfidf_arr)

errors = sum( sms['spam'] != sms['lda'] )
errors   # 0
err_rate = errors/len(mask)
err_rate        # 0   !!!
1 - err_rate    # 1   !!!

"""
Perfect fit!
Notice that we have more predictors (columns, Xs) then observations (rows).
This is WHY!

Moreover we didn't split data into test/train subsets.
But it's just to present LinearDiscriminantAnalysis.
"""

#%%
#%% topic-word matrix for LSA on 16 short sentences about cats, dogs and NYC
from nlpia.book.examples.ch04_catdog_lsa_3x6x16 import word_topic_vectors
word_topic_vectors
word_topic_vectors.T.round(1)

# compare with weights matrix W above
# still we don't know where these weights are taken from

word_topic_vectors.T @ word_topic_vectors   # almost orthonormal
word_topic_vectors.T.dot(word_topic_vectors)   # almost orthonormal

#%%