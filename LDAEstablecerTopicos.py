import csv
import os
import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Tutorial: https://blog.mlreview.com/topic-modeling-with-scikit-learn-e80d33668730
# Tutorial2: https://machinelearninggeek.com/latent-dirichlet-allocation-using-scikit-learn/
file = str(input("Introduce el path relativo (EJ: ./data/nombre.csv) :"))
nTopics = int(input("Introduce el numero de topicos (EJ: 1,2,3...):"))
alpha = float(input("Introduce el valor de alpha (EJ: 0.25, 1.0, 1.25...):"))
eta = float(input("Introduce el valor de eta (EJ: 0.25, 1.0, 1.25...):"))

df = pd.read_csv(file)
documents = df['reviewText']

from sklearn.feature_extraction.text import CountVectorizer
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
tf = tf_vectorizer.fit_transform(documents.values.astype(str))
tf_feature_names = tf_vectorizer.get_feature_names()

from sklearn.decomposition import LatentDirichletAllocation
lda_model = LatentDirichletAllocation(n_components=nTopics, doc_topic_prior=alpha, topic_word_prior=eta, max_iter=100, learning_method='online', learning_offset=50., random_state=0).fit(tf)

lda_W = lda_model.transform(tf)
print(lda_W)
lda_H = lda_model.components_ / lda_model.components_.sum(axis=1)[:, np.newaxis]
print(lda_H)

# Tutorial3: https://yanlinc.medium.com/how-to-build-a-lda-topic-model-using-from-text-601cdcbfd3a6
print(list(df))
data = df.reviewText.values.tolist()
# Create Document — Topic Matrix
lda_output = lda_model.transform(tf)
# column names
topicnames = ["Topic" + str(i) for i in range(lda_model.n_components)]
# index names
docnames = df["Unnamed: 0"]
# dominant topic for each document
# Make the pandas dataframe
df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)
# dominant topic for each document
dom_topic = np.argmax(df_document_topic.values, axis = 1)
df_document_topic["Dom_Topic"] = dom_topic

df_document_topic.to_csv("Resultados/Topics-review/MicroTMPos.csv")


