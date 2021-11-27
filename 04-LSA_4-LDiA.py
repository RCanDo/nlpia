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
subtitle: Latent Diitichlet Allocation
version: 1.0
type: tutorial
keywords: [LSA, semantic, topic, TF-IDF, LDiA]
description: |
    Latent Semantic Analysis
    Latent Diitichlet Allocation
remarks:
    - work interactively (in Spyder)
    - install NLPIA, see file 02-Tokenization.py
    -
todo:
sources:
    - title: Natural Language Processing in Action
      chapter: 04 - Finding meaning in word counts (semantic analysis)
      pages: 135-xx
      link: "D:\bib\Python\Natural Language Processing in Action.pdf"
      date: 2019
      authors:
          - fullname: Hobson Lane
          - fullname: Cole Howard
          - fullname: Hannes Max Hapke
      usage: |
          not only copy
    - link: https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation
file:
    usage:
        interactive: True   # if the file is intended to be run interactively e.g. in Spyder
        terminal: False     # if the file is intended to be run in a terminal
    name: 04-LSA_4-LDiA.py
    path: D:/Projects/Python/NLPA/
    date: 2021-10-08
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
#%%
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import casual_tokenize
np.random.seed(42)

#%% data
# from revious files
mask = sms.spam == 1

#%% create BOW
help(CountVectorizer)
counter = CountVectorizer(tokenizer=casual_tokenize)
sms_bow = pd.DataFrame(counter.fit_transform(sms.text).toarray())
sms_bow.shape    # (4837, 9232)

terms = pd.Series(counter.vocabulary_).sort_values().index.to_list()
sms_bow.columns = terms
sms.text[0]
sms_bow_0 = sms_bow.iloc[0]
sms_bow_0[sms_bow_0 > 0]

#%% LDiA fit
from sklearn.decomposition import LatentDirichletAllocation as LDiA

ldia = LDiA(n_components=16, learning_method='batch')
ldia = ldia.fit(sms_bow)
ldia.components_.shape    # (16, 9232)

#%%
topics_df = pd.DataFrame(ldia.components_.T, index=terms, columns=[f'topic{i}' for i in range(16)])
topics_df.head(22)
topics_df.shape

def topic(i, df, n=16):
    return df[f'topic{i}'].sort_values(ascending=False)[:n]

topic(3, topics_df)

#%% some nice picture
import wordcloud as wc
dir(wc)

fig, axs = plt.subplots(4, 4, figsize=(20, 16), tight_layout=True)

def wordcloud(i, df, n=16):
    wd = wc.WordCloud(width = 3000, height = 2000, background_color='black', colormap='Pastel1') \
        .generate_from_frequencies( topic(i, df, n).to_dict() )
    return wd

for i in range(4):
    for j in range(4):
        axs[i, j].imshow( wordcloud(4*i + j, topics_df) )
        axs[i, j].set_title(f'topic{4*i + j}', weight='bold', c='gray')
        axs[i, j].axis("off")

#%% How LDiA helps discriminate between spam and ham ?

#%% transform data first - LDiA transform

sms_ldia_16 = ldia.transform(sms_bow)
sms_ldia_16 = pd.DataFrame(sms_ldia_16, columns=topics_df.columns)
sms_ldia_16.head(22)

# notice that LDiA gives values between [0, 1] (unlike PCA/SVD)
sms_ldia_16.min().min()     # 9.93641407210456e-06
sms_ldia_16.max().max()     # 0.9919871750212775

#%%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(sms_ldia_16, sms.spam, test_size=.5, random_state=271828)

lda = LDA(n_components=1).fit(X_train, y_train)
lda.score(X_test, y_test)       # 0.936  good !!!

#%% do some nice plot

colors = ["blue", "green", "white","yellow", "red"]
nodes = [0, .2, .5, .8,  1]   # 'centers' of colors
cmap = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

def colormap_data(df, title):
    fig = plt.figure(figsize=[10, 17], tight_layout=True)
    ax = fig.add_subplot(111)
    mappable = ax.pcolormesh(df, cmap=cmap, rasterized=True, vmin=0, vmax=1)
    fig.colorbar(mappable, ax=ax)
    fig.suptitle(title)
    return fig, ax

colormap_data(sms_ldia_16.loc[mask, :], title="spam")
colormap_data(sms_ldia_16.loc[~mask, :], title="ham")

#%%
#%% Watch out for colinearinty!!!
""" See p. 140-141
Search first for colinearity of BOW in both dimensions! I.e.
    - colinear columns (words which only appears together)
    - colinear rows (texts consisting of exactly the same words)
"""
sms_bow_mask = sms_bow != 0
help(sms_bow_mask.duplicated)
sms_bow_dupl_rows = sms_bow_mask.duplicated()
sms_bow_dupl_rows.value_counts()        # False  4504;  True  333

sms_bow_dupl_cols = sms_bow_mask.T.duplicated()
sms_bow_dupl_cols.value_counts()        # False  5892;  True  3340

sms_bow_c = sms_bow.loc[~sms_bow_dupl_rows, ~sms_bow_dupl_cols]
sms_bow_c.shape    # (4504, 5892)

sms_c = sms.loc[~sms_bow_dupl_rows, :]
mask_c = sms_c.spam == 1

# the same for tf-idf
sms_tfidf_c = sms_tfidf_df.loc[~sms_bow_dupl_rows, ~sms_bow_dupl_cols]


#%% repeat for whole pipliene for 'new' data
ldia_16_c = LDiA(n_components=16, learning_method="batch").fit(sms_bow_c)
 # topics_df = pd.DataFrame(ldia_16_c.components_.T, index=terms, columns=[f'topic{i}' for i in range(16)])
sms_ldia_16_c = ldia_16_c.transform(sms_bow_c)
sms_ldia_16_c = pd.DataFrame(sms_ldia_16_c, columns=topics_df.columns)

X_train, X_test, y_train, y_test = train_test_split(sms_ldia_16_c, sms_c.spam, test_size=.5, random_state=271828)
lda = LDA(n_components=1).fit(X_train, y_train)
lda.score(X_test, y_test)
    # 0.954   better !!!

colormap_data(sms_ldia_16_c.loc[list(mask_c), :], title="spam")
    #!??? it doesn't want to work without list() ! WHY???
colormap_data(sms_ldia_16_c.loc[list(~mask_c), :], title="ham")

sms_ldia_16_c.loc[mask_c, :]


#%%
#%% repeat it with cross-validation

from sklearn.model_selection import cross_val_score
lda = LDA(n_components=1)
scores = cross_val_score(lda, sms_tfidf_c, sms_c.spam, cv=5)
scores
#  array([0.97 , 0.9556, 0.9478, 0.96, 0.949])

X_train, X_test, y_train, y_test = train_test_split(sms_tfidf_c, sms_c.spam, test_size=0.3, random_state=271828)
lda = LDA(n_components=1)
lda.fit(X_train, y_train)
lda.score(X_test, y_test)
#  0.959

#%% do it for LDiA
lda = LDA(n_components=1)
X_train, X_test, y_train, y_test = train_test_split(sms_ldia_16_c, sms_c.spam, test_size=0.3, random_state=271828)
lda.fit(X_train, y_train)
lda.score(X_test, y_test)
# 0.958

lda = LDA(n_components=1)
scores = cross_val_score(lda, sms_ldia_16_c, sms_c.spam, cv=5)
scores
# array([0.95, 0.959, 0.954, 0.949, 0.95])

#%%
#%%
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import numpy as np
corpus = ['this is the first document',
          'this document is the second document',
          'and this is the third one',
          'is this the first document']
vocabulary = ['this', 'document', 'first', 'is', 'second', 'the',
              'and', 'one']
pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)),
                 ('tfid', TfidfTransformer())]).fit(corpus)
pipe['count'].transform(corpus).toarray()

pipe['tfid'].idf_

pipe.transform(corpus).toarray()


#%%