import os
import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def display_topics(H, W, feature_names, documents, no_top_words, no_top_documents):
    open("ResultadosLDA.csv", 'w')  #cambiar esto para que el nombre sea aceptable
    for topic_idx, topic in enumerate(H):
        print("Topic %d:" % (topic_idx))
        print(''.join([' ' +feature_names[i] + ' ' + str(round(topic[i], 5)) #y esto también
                for i in topic.argsort()[:-no_top_words - 1:-1]]))

        top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:no_top_documents]
        docProbArray=np.argsort(W[:,topic_idx])
        print(docProbArray)
        howMany=len(docProbArray);
        print("How Many");
        print(howMany);
        for doc_index in top_doc_indices:
            print(documents[doc_index])

# Tutorial: https://blog.mlreview.com/topic-modeling-with-scikit-learn-e80d33668730
# Tutorial2: https://machinelearninggeek.com/latent-dirichlet-allocation-using-scikit-learn/
file = "HRBlockIntuitReviewsTrainDev_vLast7.csv"
df = pd.read_csv(file)
df = df.head(5000)  #a veces con más datos da fallos y no sé porque
documents = df['reviewText'].tolist()

from sklearn.feature_extraction.text import CountVectorizer
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()

from sklearn.decomposition import LatentDirichletAllocation
lda_model = LatentDirichletAllocation(n_components=100, max_iter=100, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

lda_W = lda_model.transform(tf)
#print(lda_W)
lda_H=lda_model.components_ /lda_model.components_.sum(axis=1)[:, np.newaxis]
#print(lda_H)
print("LDA Topics")
display_topics(lda_H, lda_W, tf_feature_names, documents, 10, 10)