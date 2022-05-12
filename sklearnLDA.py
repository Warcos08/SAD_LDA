import os
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Tutorial: https://blog.mlreview.com/topic-modeling-with-scikit-learn-e80d33668730
# Tutorial2: https://machinelearninggeek.com/latent-dirichlet-allocation-using-scikit-learn/
df = pd.read_csv("./HRBlockIntuitReviewsTrainDev_vLast7.csv")
df = df.head(5000)  #a veces con más datos da fallos y no sé porque
documents = df['reviewText'].tolist()

from sklearn.feature_extraction.text import CountVectorizer
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words="english")
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()
print(tf_feature_names)

from sklearn.decomposition import LatentDirichletAllocation
lda_model = LatentDirichletAllocation(n_components=100, max_iter=5, learning_method='online', learning_offset=50., random_state=0).fit(tf)

lda_W = lda_model.transform(tf)
lda_H = lda_model.components_

# Imprimimos los tópicos necesarios
print("LDA Topics:")
terms = tf_vectorizer.get_feature_names()
for index, component in enumerate(lda_H):
    zipped = zip(terms, component)
    top_terms_key=sorted(zipped, key = lambda t: t[1], reverse=True)[:7]
    top_terms_list=list(dict(top_terms_key).keys())
    print("Topic "+str(index)+": ",top_terms_list)