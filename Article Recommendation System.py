#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


data=pd.read_csv('C:/Users/Rakesh/Datasets/articles.csv', encoding='latin1')


# In[3]:


data.head()


# In[7]:


articles = data['Article'].tolist()
uni_tfidf = text.TfidfVectorizer(input=articles, stop_words='english')
uni_matrix = uni_tfidf.fit_transform(articles)
uni_sim = cosine_similarity(uni_matrix)


# In[8]:


def recommend_articles(x):
    return", ".join(data['Title'].loc[x.argsort()[-5:-1]])
data['Recommended Articles'] = [recommend_articles(x) for x in uni_sim]
data.head()


# ## As you can see from the output above, a new column has been added to the dataset that contains the titles of all the recommended articles. Now let’s see all the recommendations for an article

# In[10]:


print(data["Recommended Articles"][2])

