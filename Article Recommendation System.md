```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
```


```python
data=pd.read_csv('C:/Users/Rakesh/Datasets/articles.csv', encoding='latin1')
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Article</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Data analysis is the process of inspecting and...</td>
      <td>Best Books to Learn Data Analysis</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The performance of a machine learning algorith...</td>
      <td>Assumptions of Machine Learning Algorithms</td>
    </tr>
    <tr>
      <th>2</th>
      <td>You must have seen the news divided into categ...</td>
      <td>News Classification with Machine Learning</td>
    </tr>
    <tr>
      <th>3</th>
      <td>When there are only two classes in a classific...</td>
      <td>Multiclass Classification Algorithms in Machin...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The Multinomial Naive Bayes is one of the vari...</td>
      <td>Multinomial Naive Bayes in Machine Learning</td>
    </tr>
  </tbody>
</table>
</div>




```python
articles = data['Article'].tolist()
uni_tfidf = text.TfidfVectorizer(input=articles, stop_words='english')
uni_matrix = uni_tfidf.fit_transform(articles)
uni_sim = cosine_similarity(uni_matrix)
```


```python
def recommend_articles(x):
    return", ".join(data['Title'].loc[x.argsort()[-5:-1]])
data['Recommended Articles'] = [recommend_articles(x) for x in uni_sim]
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Article</th>
      <th>Title</th>
      <th>Recommended Articles</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Data analysis is the process of inspecting and...</td>
      <td>Best Books to Learn Data Analysis</td>
      <td>Introduction to Recommendation Systems, Best B...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The performance of a machine learning algorith...</td>
      <td>Assumptions of Machine Learning Algorithms</td>
      <td>Applications of Deep Learning, Best Books to L...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>You must have seen the news divided into categ...</td>
      <td>News Classification with Machine Learning</td>
      <td>Language Detection with Machine Learning, Appl...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>When there are only two classes in a classific...</td>
      <td>Multiclass Classification Algorithms in Machin...</td>
      <td>Assumptions of Machine Learning Algorithms, Be...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The Multinomial Naive Bayes is one of the vari...</td>
      <td>Multinomial Naive Bayes in Machine Learning</td>
      <td>Assumptions of Machine Learning Algorithms, Me...</td>
    </tr>
  </tbody>
</table>
</div>



## As you can see from the output above, a new column has been added to the dataset that contains the titles of all the recommended articles. Now let’s see all the recommendations for an article


```python
print(data["Recommended Articles"][2])
```

    Language Detection with Machine Learning, Apple Stock Price Prediction with Machine Learning, Multiclass Classification Algorithms in Machine Learning, News Classification with Machine Learning
    
