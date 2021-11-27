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
keywords: [LSA, semantic, topic, TF-IDF, SVD]
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
      pages: 116-xx
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
    name: 04-LSA_2.py
    path: D:/Projects/Python/NLPA/
    date: 2021-09-29
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

Then we may reduce dim taking only p largest singular values for some p <= min(m, n):
W' = U' S' V'*   (m x n) = (m x p) * p2 * (p x n)
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

Notice that comparing with (1) (previous file) we have analogy:
Vt' ~= T  (p x n)  topic-document
Ut' ~= M  (p x m)  topic-word  weights

"""

#%%
from nlpia.book.examples.ch04_catdog_lsa_sorted import lsa_models, prettify_tdm

# LSA on cats_and_dogs corpus
bow_svd, tfidf_svd = lsa_models()
bow_svd

prettify_tdm(**bow_svd)

bow_svd['tdm']          # (m x n)  word-doc  - BOW (bag of words)
bow_svd['u'].round(2)   # m2       word-topic
bow_svd['s'].round(2)   # r = min(m, n)   topic variations; just vector - may be transform to (m x n) as below (*)
bow_svd['vt'].round(2)  # n2       topic-doc   -- 'd' in index suggests it's for 'document' but it's not!
                                                # should be rather `t` as each row is for 'topic'.
                                                # documents goes with columns
bow_svd['docs']         # corpus

#%%
#%% !!! SVD  via  NumPy !!!
import numpy as np
U, s, Vt = np.linalg.svd(bow_svd['tdm'])
U.round(2)  #
s.round(2)
Vt.round(2)

#%% (*)
S = np.zeros((len(U), len(Vt)))       # U, V  are square matrices
np.fill_diagonal(S, s)                #! inplace !!! ...
S.round(2)

#%%
W = U @ S @ Vt
W.round(1)
bow_svd['tdm']           #!!! OK !!!

#%% using Einstein notation
np.einsum('ij, jk -> ik', U, S)
np.einsum('ij, jk, kl -> il', U, S, Vt).round(1)
#!!! GREAT !!!

#%% Term-document matrix reconstruction error, p.122
errs = []
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

#%%
#%%
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize

tfidf = TfidfVectorizer(tokenizer = casual_tokenize)

tfidf.fit(raw_documents = bow_svd['docs'])
tfidf.vocabulary_

tfidf_arr = tfidf.transform(raw_documents = bow_svd['docs']).toarray()
tfidf_arr.round(2)

#%%
#%% how is tfidf.vocabulary_ sorted? (anyhow?)

from functools import reduce, partial

corpus = bow_svd['docs'].tolist()
corpus

def repl(string: str, noise: str):
    """remove every sign in the `noise` from the `string`"""
    return reduce(lambda s, z: s.replace(z, ""), list(noise), string)
# repl("a!bc$d-ef_g", "!$-_,?")   # 'przsab'

docs = [d.lower() for d in map(partial(repl, noise="!?,.()"),  corpus)]
docs

#%% in one line (horrible to understand! just as excercise)
[d.lower() for d in map(lambda s: reduce(lambda s, z: s.replace(z, ""), list("!?,.()"), s),  corpus)]

#%%
from collections import Counter

counter = Counter()

for d in docs:
    counter.update(d.split())
counter.most_common()

#%% compare it with tfidf
tfidf.vocabulary_

"""
NO APPARENT RELATION;
the number here is only the index of a respective column in the final TF-IDF table;
and this index is apparently not related with the word frequency in the corpus:
"""
tfidf_arr.round(2)

#%%
""" !!! IT'S TERRIBLY UNREADABLE !!!
Why it's NOT sorted ???
"""
def to_df(arr, index_dict):
    df = pd.DataFrame( arr.round(2),
              columns = pd.Series(index_dict).sort_values().index
            ).T
    return df

print_df(to_df(tfidf_arr, tfidf.vocabulary_))

#%% make model without all the stop-words and punctuation;
# vocab is limited here so we can just pass it to `vocabulary` argument
# we limit it only to words from the example
bow_svd['tdm']

tfidf_0 = TfidfVectorizer(tokenizer = casual_tokenize, vocabulary="cat dog apple lion nyc love".split())
tfidf_0.fit(raw_documents=bow_svd['docs'])
tfidf_0.vocabulary_

tfidf_0_arr = tfidf_0.transform(raw_documents=bow_svd['docs']).toarray()

print_df(to_df(tfidf_0_arr, tfidf_0.vocabulary_))

#%%
U0, s0, Vt0 = np.linalg.svd(tfidf_0_arr.T)

S0 = np.zeros((len(U0), len(Vt0)))
np.fill_diagonal(S0, s0)

(U0 @ S0 @ Vt0).round(2)
tfidf_0_arr.round(2)

#%% Term-document matrix reconstruction error, p.122
errs0 = []
S0prim = S0.copy()
for d in range(len(s), 0, -1):  # print(d)
    print(S0prim.round(2))
    W0prim = U0 @ S0prim @ Vt0
    err = np.sqrt( sum((tfidf_0_arr.T - W0prim).flatten() ** 2) )
    print(err)
    errs0 += [err]
    if d > 0:
        S0prim[d-1, d-1] = 0
print([round(x, 2) for x in errs0])

#%%
plt.plot(errs)
plt.plot(errs0)

#%%
