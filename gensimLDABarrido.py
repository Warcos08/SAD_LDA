# Parte 1
import json
import re
import pandas as pd
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import ToktokTokenizer
import csv

# Parte 2
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import random
import numpy as np
import matplotlib.pyplot as plt
#from wordcloud import WordCloud

STOPWORDS = set(stopwords.words("english"))
wnl = WordNetLemmatizer()
stemmer = SnowballStemmer("english")

def limpiar_texto(texto):
    # Eliminamos los caracteres especiales
    texto = re.sub(r'\W', ' ', str(texto))
    # Eliminado las palabras que tengo un solo caracter
    texto = re.sub(r'\s+[a-zA-Z]\s+', ' ', texto)
    # Sustituir los espacios en blanco en uno solo
    texto = re.sub(r'\s+', ' ', texto, flags=re.I)
    # Convertimos textos a minusculas
    texto = texto.lower()
    return texto

def eliminar_stopwords(tokens):
    return [token for token in tokens if token not in STOPWORDS and not token.isdigit()]

def lematizar(tokens):
    return [wnl.lemmatize(token) for token in tokens]

def estemizar(tokens):
    return [stemmer.stem(token) for token in tokens]


# ---> Parte 1: https://elmundodelosdatos.com/topic-modeling-gensim-fundamentos-preprocesamiento-textos/
ruta = str(input("Introduce el path relativo (EJ: ./data/nombre.csv) :"))
df = pd.read_csv(ruta)
df = df[['reviewText', 'summary']]

# 1.- Limpiamos (quitar caracteres especiaes, minúsculas...)
df["Tokens"] = df.reviewText.apply(limpiar_texto)

# 2.- Tokenizamos
tokenizer= ToktokTokenizer()
df["Tokens"] = df.Tokens.apply(tokenizer.tokenize)

# 3.- Eliminar stopwords y digitos
df["Tokens"] = df.Tokens.apply(eliminar_stopwords)

# 4.- ESTEMIZAR / LEMATIZAR ???
df["Tokens"] = df.Tokens.apply(estemizar)
print(df.Tokens[0][0:10])



# ---> Parte 2: https://elmundodelosdatos.com/topic-modeling-gensim-asignacion-topicos/
# Cargamos en el diccionario la lista de palabras que tenemos de las reviews
diccionario = Dictionary(df.Tokens)
print(f'Número de tokens: {len(diccionario)}') #mostrar el numero se palabras

# Reducimos el diccionario filtrando las palabras mas raras o demasiado frecuentes
# no_below = mantener tokens que se encuentran en el a menos x documentos
# no_above = mantener tokens que se encuentran en no mas del 80% de los documentos
diccionario.filter_extremes(no_below=2, no_above = 0.8)
print(f'Número de tokens: {len(diccionario)}')

# Creamos el corpus (por cada roken en el df) QUE ES UN ARRAY BOW
corpus = [diccionario.doc2bow(review) for review in df.Tokens]

# BOW de una review
print(corpus[5])

cabeceras = ["num_topics", "alpha", "beta"]
nombreCSV="GensimParams" + ruta.split('/')[-1].split('/')[-1]
archivo = open(nombreCSV, "w")
writer = csv.writer(archivo)
writer.writerow(cabeceras)
archivo.close()
print("PREPARANDO ARCHIVO .CSV PARA VOLCAR PARAMETROS...")

for t in range(0, 2):
    if t == 0:
        tipo = 'auto'
    elif t == 1:
        tipo = 'symmetric'


    for nTopics in range (10, 101, 5):

        print("------------------------------")
        print("alpha and beta --> " + tipo)
        print("n_topics --> " + str(nTopics))
        print("------------------------------")

        lda = LdaModel(corpus=corpus, id2word=diccionario,
                       num_topics=nTopics, random_state=42,
                       chunksize=1000, passes=10,
                       alpha=tipo, eta=tipo)

        # Imprimimos los topicos creados con las 5 palabras que más contribuyen a ese tópico y sus pesos
        topicos = lda.print_topics(num_words=5, num_topics=nTopics)
        for topico in topicos:
            print(topico)

        print('ALPHA -->' + str(lda.alpha))
        print('ETA -->' + str(lda.eta))

        #hay q sacar el valor de alpha y beta de alguna manera y lode symetric es sustituirlo por auto
        archivo = open(nombreCSV, "a")
        contenido = [str(nTopics), str(lda.alpha), str(lda.eta)]
        writer = csv.writer(archivo)
        writer.writerow(contenido)  # se escribe cuando el array se completa
        archivo.close()
'''
# Nube de palabras, donde se ven las palbras de los topicos con un tamaño equivalente a su relevancia en el documento
for i in range(1, 5):
    plt.figure()
    plt.imshow(WordCloud(background_color='white', prefer_horizontal=1.0)
               .fit_words(dict(lda.show_topic(i, 20))))
    plt.axis("off")
    plt.title("Tópico " + str(i))
    plt.show()
'''

# Aqui imprimimos una review aleatoria para comprobar la eficacia de nuestro modelo
indice_review = random.randint(0,len(df))
review = df.iloc[indice_review]

print("***********************")
print("\nReview: " + review[0] + "\n")
print("***********************")

# Obtenemos el BOW de la review
# Obtenemos la distribucion de topicos
bow_review = corpus[indice_review]
distribucion_review = lda[bow_review]

# Indices de los topicos mas significativos
dist_indices = [topico[0] for topico in lda[bow_review]]
# Contribución de los topicos mas significativos
dist_contrib = [topico[1] for topico in lda[bow_review]]

# Representacion grafica de los topicos mas significativos
distribucion_topicos = pd.DataFrame({'Topico':dist_indices,
                                     'Contribucion':dist_contrib })
distribucion_topicos.sort_values('Contribucion',
                                 ascending=False, inplace=True)
ax = distribucion_topicos.plot.bar(y='Contribucion',x='Topico',
                                   rot=0, color="orange",
                                   title = 'Tópicos mas importantes'
                                   'de review ' + str(indice_review))

plt.show()

# Imprimimos las palabras mas significativas de los topicos
for ind, topico in distribucion_topicos.iterrows():
    print("*** Tópico: " + str(int(topico.Topico)) + " ***")
    palabras = [palabra[0] for palabra in lda.show_topic(
        topicid=int(topico.Topico))]
    palabras = ', '.join(palabras)
    print(palabras, "\n")

"""
# Podemos incluir una nueva review para probar el modelo
texto_review = open("review.txt")
review_nuevo = texto_review.read().replace("\n", " ")
texto_review.close()


# BOW de la nueva review
review_nuevo = limpiar_texto(review_nuevo)
review_nuevo = tokenizer.tokenize(review_nuevo)
review_nuevo = eliminar_stopwords(review_nuevo)
review_nuevo = estemizar(review_nuevo)

bow_review_nuevo = diccionario.doc2bow(review_nuevo)

# Mostramos los resultados como antes
distribucion_topicos = pd.DataFrame({'Topico':dist_indices,
                                     'Contribucion':dist_contrib })
distribucion_topicos.sort_values('Contribucion', 
                                 ascending=False, inplace=True)
ax = distribucion_topicos.plot.bar(y='Contribucion',x='Topico', 
                                   rot=0, color="green",
                                   title = 'Tópicos más importantes' 
                                   'para documento nuevo')

# Imprimimos de nuevo las palabras mas significativas
for ind, topico in distribucion_topicos.iterrows():
    print("*** Tópico: " + str(int(topico.Topico)) + " ***")
    palabras = [palabra[0] for palabra in lda.show_topic(
        topicid=int(topico.Topico))]
    palabras = ', '.join(palabras)
    print(palabras, "\n")
"""

# Guardamos el modelo y el diccionario para usarlo de nuevo mas adelante
# lda.save("review.model")
# diccionario.save("review.dictionary")

