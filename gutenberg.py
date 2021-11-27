# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 08:42:51 2021

@author: staar

https://www.gutenberg.org/ebooks/search/?query=fairy+tales&submit_search=Go%21
"""

#%%
import requests
import os

#%%
DATA_PATH = "E:/Data/nlp/books"

gutenberg_fairy_tales = \
{
0:
    {
    "title": "The Children Of Odin",
    "author": "Padraic Colum",
    "date": "2008-03-02",
    "encoding": "ascii",
    "url": "https://www.gutenberg.org/files/24737/24737.txt",
    "language": "en",
    "structure": {
        "toc": 145,
        "start": 244,
        "end": 7231,
        }
    },
1:
    {
    "title": "Favorite Fairy Tales",
    "author": "Logan Marshall",
    "date": "2007-03-16",
    "encoding": "ascii",
    "url": "https://www.gutenberg.org/files/20748/20748.txt",
    "language": "en",
    "structure": {
        "toc": 65,
        "start": 108,
        "end": 4633,
        }
    },
2:
    {
    "title": "The Blue Fairy Book",
    "author": "Various",
    "date": "2009-11-30",
    "encoding": "ascii",
    "url": "https://www.gutenberg.org/files/503/503.txt",
    "language": "en",
    "structure": {
        "toc": 42,
        "start": 86,
        "end": 13567,
        }
    },
3:
    {
    "title": "Wonder Tales from Tibet",
    "author": "Eleanore Myers Jewett",
    "date": "2021-09-01",
    "encoding": "ascii",
    "url": "https://www.gutenberg.org/files/66443/66443-0.txt",
    "language": "en",
    "structure": {
        "toc": 109,
        "start": 155,
        "end": 3298,
        }
    },
}

#%%
def write(id: int, data=gutenberg_fairy_tales, path=DATA_PATH):

    import requests

    try:
        item = data[id]
        txt_file = f"{item['author']} - {item['title']}.txt"
        file_path = os.path.join(path, txt_file)

        r = requests.get(url = item["url"], allow_redirects=True)
        open(file_path, 'bw').write(r.content)

        return True

    except Exception as e:
        print(e)
        return False

#%%
write(3)


#%%
#%%
from nltk.corpus import gutenberg
gutenberg.fileids()
"""
['austen-emma.txt',
 'austen-persuasion.txt',
 'austen-sense.txt',
 'bible-kjv.txt',
 'blake-poems.txt',
 'bryant-stories.txt',
 'burgess-busterbrown.txt',
 'carroll-alice.txt',
 'chesterton-ball.txt',
 'chesterton-brown.txt',
 'chesterton-thursday.txt',
 'edgeworth-parents.txt',
 'melville-moby_dick.txt',
 'milton-paradise.txt',
 'shakespeare-caesar.txt',
 'shakespeare-hamlet.txt',
 'shakespeare-macbeth.txt',
 'whitman-leaves.txt']
"""
dir(gutenberg)
help(gutenberg)

#%%
#%%