#! python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: Xxx Yy Zzz
subtitle:
version: 1.0
type: tutorial
keywords: [word2vec, neural network, word embedding]
description: |
    Shortly about content ...
remarks:
    - work interactively (in Spyder)
    - install NLPIA, see file 02-Tokenization.py
    -
todo:
sources:
    - title: Natural Language Processing in Action
      chapter: 06 - Reasoning with Word2vec
      pages: 181-xx
      link: "D:\bib\Python\Natural Language Processing in Action.pdf"
      date: 2019
      authors:
          - fullname: Hobson Lane
          - fullname: Cole Howard
          - fullname: Hannes Max Hapke
      usage: |
          not only copy
    - title: Google's word2vec
      link: https://code.google.com/archive/p/word2vec/
    - title: model data
      link: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
    - title: the same via git
      link: https://github.com/mmihaltz/word2vec-GoogleNews-vectors
file:
    usage:
        interactive: True   # if the file is intended to be run interactively e.g. in Spyder
        terminal: False     # if the file is intended to be run in a terminal
    name: 06-Word2Vec.py
    path: D:/Projects/Python/NLPA/
    date: 2020-02-13
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

WD = os.getcwd()
print(WD)

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
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('dark_background')
# see `plt.style.available` for list of available styles

from mpl_toolkits.mplot3d import Axes3D

from matplotlib.colors import LinearSegmentedColormap

import seaborn as sn

#%%
pd.options.display.width = 120

#%%
#%%
#%% p.186
#! DO NOT LOAD IT - IT'S HUGE - pretrained word2vec model
from nlpia.book.examples.ch06_nessvectors import *
nessvector('Marie_Curie').round(2)

#%% p.200
#! this is also HUGE (and not used below)
from nlpia.data.loaders import get_data
word_vectors = get_data('word2vec')

#%%
# page:  https://code.google.com/archive/p/word2vec/
# data:  https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
from gensim.models.keyedvectors import KeyedVectors
word_vectors = KeyedVectors.load_word2vec_format(
        'E:/Data/nlp/GoogleNews-vectors-negative300.bin.gz',
        binary=True, limit=200000)

#%%

word_vectors.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
# [('queen', 0.7118192911148071)]

word_vectors.similarity('queen', 'princess')    # 0.7070532
word_vectors.similarity('dog', 'cat')           # 0.76094574
word_vectors.similarity('mouse', 'cat')         # 0.46566278

word_vectors.most_similar(positive=['mouse', 'dog'], negative=['cat'], topn=1)
# [('cursor', 0.4924313724040985)]

word_vectors.most_similar(positive=['Warsaw'], topn=1)   # [('Prague', 0.6626689434051514)]

word_vectors.most_similar(positive=['cooking', 'potatoes'], topn=5)
"""
[('cook', 0.6973530054092407),
 ('sweet_potatoes', 0.6600279808044434),
 ('vegetables', 0.6513738036155701),
 ('onions', 0.6512383222579956),
 ('baking', 0.6481684446334839)]
"""

word_vectors.doesnt_match("potatoes milk cake computer".split())
# 'computer'

#%%
word_vectors['phone']   # array([...], )
len(word_vectors['phone'])   # 300

#%%
#%% p.203
corpus     # <- load from somwhere ...
# sentence segmenter, e.g. Detector Morse -> https://github.com/cslu-nlp/DetectorMorse
# ...
# apply tokenizer, stemmer, lemmatizer ... (as in ch2)
# ->  e.g.
token_list = [ "to provide early intervention childhood special education services to eligible children and their families".split(),
               "essential job function".split(),
               "participate as a transdisciplinary team member to complete educational assessments for".split(),
               ...
             ]
# list of lists of tokens

#%%
from gensim.models.word2vec import Word2Vec

num_features = 300      # nr of dimensions to represent word vector
min_word_count = 3      # min nr of word occurences for word to be considered
num_workers = 2         # CPU cores to be used; or:  import multiprocessing \ num_workers = multiprocessing.cpu_count()
window_size = 6         # context window size
subsampling = 1e-3      # subsampling rate for frequent terms, p.198

#%%
model = Word2Vec(
    token_list,
    workers = num_works,
    size = num_features,
    min_count = min_word_count,
    window = window_size,
    sample = subsampling
    )

#%% only the weight matrix for the hidden layer is of interest
model.init_sims(replace=True)
#

#%%


#%%


#%%