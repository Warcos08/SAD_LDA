from gensim.corpora import Dictionary
from gensim.models import LdaModel
import random
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# https://elmundodelosdatos.com/topic-modeling-gensim-asignacion-topicos/

df = "dataFrame"
# Cargamos en el diccionario la lista de palabras que tenemos de las reviews
diccionario = Dictionary(df.Tokens)
print(f'Número de tokens: {len(diccionario)}')

# Reducimos el diccionario filtrando las palabras mas raras o demasiado frecuentes
# no_below = mantener tokens que se encuentran en el a menos x documentos
# no_above = mantener tokens que se encuentran en no mas del 80% de los documentos
diccionario.filter_extremes(no_below=2, no_above = 0.8)
print(f'Número de tokens: {len(diccionario)}')

# Creamos el corpus
corpus = [diccionario.doc2bow(review) for review in df.Tokens]

# BOW de una review
print(corpus[5])

lda = LdaModel(corpus=corpus, id2word=diccionario,
               num_topics=50, random_state=42,
               chunksize=1000, passes=10, alpha='auto')

# Imprimimos los topicos creados con las 5 palabras que más contribuyen a ese tópico y sus pesos
topicos = lda.print_topics(num_words=5, num_topics=50)
for topico in topicos:
    print(topico)

# Nube de palabras, donde se ven las palbras de los topicos con un tamaño equivalente a su relevancia en el documento
for i in range(1, 5):
    plt.figure()
    plt.imshow(WordCloud(background_color='white', prefer_horizontal=1.0)
               .fit_words(dict(lda.show_topic(i, 20))))
    plt.axis("off")
    plt.title("Tópico " + str(i))
    plt.show()

# Aqui imprimimos una review aleatoria para comprobar la eficacia de nuestro modelo
"""
indice_review = random.randint(0,len(df))
review = df.iloc[indice_review]
print(str(review.compañia) + ": ")
print(review.review)
"""




