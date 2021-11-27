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
subtitle: Principal Component Analysis
version: 1.0
type: tutorial
keywords: [LSA, semantic, topic, TF-IDF, PCA]
description: |
    Latent Semantic Analysis
    Principal Component Analysis
remarks:
    - work interactively (in Spyder)
    - install NLPIA, see file 02-Tokenization.py
    -
todo:
sources:
    - title: Natural Language Processing in Action
      chapter: 04 - Finding meaning in word counts (semantic analysis)
      pages: 125-xx
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
    name: 04-LSA_3-PCA.py
    path: D:/Projects/Python/NLPA/
    date: 2021-10-07
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
#%% 3-d data
#%%
from nlpia.data.loaders import get_data
X = get_data('pointcloud').sample(1000)
X.shape
X.head()

#%% 3-d plot
fig = plt.figure(figsize=[9, 7], tight_layout=True)
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('gray')
ax.scatter(X.x, X.y, X.z, c='k', s=2)

#%% PCA -- all components
from sklearn.decomposition import PCA
pca = PCA()  #n_components=2)
pca_fit = pca.fit(X)
dir(pca_fit)
vars(pca_fit)

#%% 2 components
pca_fit_2 = PCA(n_components=2).fit(X)
vars(pca_fit_2)
# 'noise' are all components which were rejected

X2 = pd.DataFrame(pca_fit_2.transform(X), columns=['x', 'y'])

plt.scatter(X2.x, X2.y, c='r', s=2)

#%% some check
pca_fit_2.components_.T  # right eigenvectors of SVD of Cov(X)

X2_ = X @ pca_fit_2.components_.T
X2_.columns = list('xy')

plt.figure(2)
plt.scatter(X2_.x, X2_.y, c='r', s=2)   ## the same !!!

#%% sms data
#%%
sms = get_data('sms-spam')
sms.head()
sms.spam.sum()    # 638
mask = sms.spam == 1

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize

tfidf = TfidfVectorizer(tokenizer=casual_tokenize)
tfidf.vocabulary_

sorted(tfidf.vocabulary_.items(), key=lambda x: x[1])[:100]

sms_tfidf = tfidf.fit_transform(raw_documents=sms.text)
sms_tfidf   # <4837x9232 sparse matrix of type '<class 'numpy.float64'>' with 82353 stored elements in Compressed Sparse Row format>
sms_tfidf.shape     # (4837, 9232)

sms_tfidf_df = pd.DataFrame(sms_tfidf.toarray())
terms = pd.Series(tfidf.vocabulary_).sort_values().index.to_list()
sms_tfidf_df.columns = terms

sms_tfidf_df = sms_tfidf_df - sms_tfidf_df.mean()

#%% PCA - fit
from sklearn.decomposition import PCA

TOPICS = 16

pca = PCA(n_components=TOPICS).fit(sms_tfidf_df)
pca.components_
pca.components_.shape   # (16, 9232)

#%% PCA - transform
sms_topics = pca.transform(sms_tfidf_df)
sms_topics = pd.DataFrame(sms_topics, columns=[f'topic{i}' for i in range(TOPICS)])
sms_topics.shape        # (4837, 16)
sms_topics.head()

sms_topics.loc[mask, :].head()

#%% how much each topic "contributes" to given document
colors = ["blue", "green", "white","yellow", "red"]
nodes = [0, .45, .5, .55,  1]   # 'centers' of colors
cmap = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

def colormap_data(df, title):
    fig = plt.figure(figsize=[10, 17], tight_layout=True)
    ax = fig.add_subplot(111)
    mappable = ax.pcolormesh(df, cmap=cmap, rasterized=True, vmin=-1, vmax=1)
    fig.colorbar(mappable, ax=ax)
    fig.suptitle(title)
    return fig, ax

colormap_data(sms_topics.loc[mask, :], title="spam")
colormap_data(sms_topics.loc[~mask, :], title="ham")

#%%
# how much each token contributes to the given topic
# not readable
#
topic_weights = pd.DataFrame(
    pca.components_.T,
    columns = [f'topic{i}' for i in range(TOPICS)],
    index = pd.Series(tfidf.vocabulary_).sort_values().index
)
topic_weights.head(100).round(4)

#%% for each topic find most important tokens
from functools import reduce

topic = lambda i: pd.DataFrame(list(topic_weights[f'topic{i}'].sort_values(ascending=False)[:16].round(3).to_dict().items()))
add_topic = lambda df, i: pd.concat((df, topic(i)), axis=1)
topics_df = reduce(add_topic, range(TOPICS), pd.DataFrame())

midx = pd.MultiIndex.from_product([[f'topic{i}' for i in range(TOPICS)], ['token', 'tfidf']], names=['topic', 'info'])
topics_df.columns = midx

topics_df

#%%
topics_df.loc[:, (slice(None), 'token')]
# or
idx = pd.IndexSlice
topics_df.loc[:, idx[:, 'token']]
topics_df.loc[:, idx[:, 'tfidf']]
topics_df['topic0']

dict(list( tuple(i) for i in topics_df['topic0'].values))

#%% some nice plot
import wordcloud as wc
dir(wc)

fig, axs = plt.subplots(4, 4, figsize=(20, 16), tight_layout=True)

def wordcloud(i):
    wd = wc.WordCloud(width = 3000, height = 2000, background_color='black', colormap='Pastel1') \
        .generate_from_frequencies( dict(list( tuple(r) for r in topics_df[f'topic{i}'].values)) )
    return wd

for i in range(4):
    for j in range(4):
        axs[i, j].imshow( wordcloud(4*i + j) )
        axs[i, j].set_title(f'topic{4*i + j}', weight='bold')
        axs[i, j].axis("off")

#%% LDA on PCA-transformed data
TODO !!!
...


#%% truncated SVD
#%%
""" TruncatedSVD() works on sparse data, so it's very good choice for large corpuse
Results should be almost the same as for PCA()
"""
from sklearn.decomposition import TruncatedSVD

sms_tfidf   # <4837x9232 sparse matrix of type '<class 'numpy.float64'>' with 82353 stored elements in Compressed Sparse Row format>

# NON-CENTERED data !!!  PCA centers data (works on covariance matrix!)

svd = TruncatedSVD(n_components=TOPICS, n_iter=100).fit(sms_tfidf)
svd.components_

sms_topics_svd = svd.transform(sms_tfidf)
sms_topics_svd = pd.DataFrame(sms_topics_svd, columns=[f'topic{i}' for i in range(TOPICS)])
sms_topics_svd.head()
# compare with PCA result
sms_topics.head()
# THEY DIFFER !!! because data were not centered

#%% now on centered data:
type(sms_tfidf_df)    # this is data frame - not a sparse matrix
svd = TruncatedSVD(n_components=TOPICS, n_iter=100).fit(sms_tfidf_df)
# works much longer...

sms_topics_svd = svd.transform(sms_tfidf_df)
sms_topics_svd = pd.DataFrame(sms_topics_svd, columns=[f'topic{i}' for i in range(TOPICS)])
sms_topics_svd.head()
# compare with PCA result
sms_topics.head()

# still different ???

#%%

#%%


#%%