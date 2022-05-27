import csv
import os
import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def display_topics(H, W, feature_names, documents, no_top_words, no_top_documents):
    #cabeceras = ["num_topics_Total", "alpha", "beta", "topicoAct", "words"]
    #nombreCSV="Resultados/LDAParams-" + file.split('/')[-1].split('/')[-1]
    #archivo = open(nombreCSV, "w")
    #writer = csv.writer(archivo)
    #writer.writerow(cabeceras)
    #archivo.close()
    #print("PREPARANDO ARCHIVO .CSV PARA VOLCAR PARAMETROS...")
    #archivo = open(nombreCSV, "a")

    archivo = open('Resultados/txt/LDA-' + str(nTopics) + '-' + str(alpha) + '-' + str(eta) + '-SonyNeg.txt', 'w')   #PARA QUE SALGA EN FORMATO TXT

    for topic_idx, topic in enumerate(H):
        print("Topic %d:" % (topic_idx))
        archivo.write("\nTopic %d:" % (topic_idx))
        print(''.join([' ' + feature_names[i] + ' ' + str(round(topic[i], 5)) #y esto también
                for i in topic.argsort()[:-no_top_words - 1:-1]]))
        archivo.write(''.join([' ' + feature_names[i] + ' ' + str(round(topic[i], 5))
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

        top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:no_top_documents]
        docProbArray=np.argsort(W[:,topic_idx])
        print(docProbArray)
        howMany=len(docProbArray);
        print("\nHow Many");
        print(howMany);
        archivo.write("\n How Many: " + str(howMany) +"\n")

        #contenido = [str(nTopics), str(alpha), str(eta), topic_idx, infoTopics]
        #writer = csv.writer(archivo)
        #writer.writerow(contenido)

        for doc_index in top_doc_indices:
            print(documents[doc_index])
            archivo.write(str(documents[doc_index]))
            archivo.write("\n")
    archivo.close()

# Tutorial: https://blog.mlreview.com/topic-modeling-with-scikit-learn-e80d33668730
# Tutorial2: https://machinelearninggeek.com/latent-dirichlet-allocation-using-scikit-learn/
file = str(input("Introduce el path relativo (EJ: ./data/nombre.csv) :"))
nTopics = int(input("Introduce el numero de topicos (EJ: 1,2,3...):"))
alpha = float(input("Introduce el valor de alpha (EJ: 0.25, 1.0, 1.25...):"))
eta = float(input("Introduce el valor de eta (EJ: 0.25, 1.0, 1.25...):"))

#file = "HRBlockIntuitReviewsTrainDev_vLast7.csv"
df = pd.read_csv(file)
#df = df.sample(n=6000, random_state=1)  #a veces con más datos da fallos y no sé porque
documents = df['reviewText']

from sklearn.feature_extraction.text import CountVectorizer
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
tf = tf_vectorizer.fit_transform(documents.values.astype(str))
tf_feature_names = tf_vectorizer.get_feature_names()

from sklearn.decomposition import LatentDirichletAllocation
lda_model = LatentDirichletAllocation(n_components=nTopics, doc_topic_prior=alpha, topic_word_prior=eta, max_iter=100, learning_method='online', learning_offset=50., random_state=0).fit(tf)

lda_W = lda_model.transform(tf)
#print(lda_W)
lda_H = lda_model.components_ / lda_model.components_.sum(axis=1)[:, np.newaxis]
#print(lda_H)
print("LDA Topics")
#print(tf_feature_names)
display_topics(lda_H, lda_W, tf_feature_names, documents, 10, 10)
