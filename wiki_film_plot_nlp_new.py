from __future__ import print_function
import numpy as np
import pandas as pd
import nltk
# nltk.download('stopwords')
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3

# change to your own path
movies_df = pd.read_csv('~/Dropbox/ImmInnHollywood/data/wiki_film_plot/wiki_movie_plots_deduped.csv')
# print(movies_df.head())

movies_df_sub = movies_df[(movies_df['Origin/Ethnicity']=="American")].reset_index(drop=True)
# movies_df_sub = movies_df[(movies_df['Release Year']=='2000')
                          # & (movies_df['Origin/Ethnicity']=="American")][1:10].reset_index(drop=True)
# print(movies_df_sub['Title'][:10])
# print
# print

title = movies_df_sub['Title']
# print(title)
release_year = movies_df_sub['Release Year']
director = movies_df_sub['Director']
genre = movies_df_sub['Genre']
plot = movies_df_sub['Plot']

stopwords = nltk.corpus.stopwords.words('english')

# print(stopwords[:10])

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens
totalvocab_stemmed = []
totalvocab_tokenized = []
for i in plot:
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list

    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)


vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index=totalvocab_stemmed)
print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')

# print(vocab_frame.head())
# print
# print
# print
# print


from sklearn.feature_extraction.text import TfidfVectorizer

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                   min_df=0.2, stop_words='english',
                                   use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 3))

tfidf_matrix = tfidf_vectorizer.fit_transform(plot) #fit the vectorizer to synopses

# print(tfidf_matrix.shape)

terms = tfidf_vectorizer.get_feature_names()

from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)
print
print

from sklearn.cluster import KMeans
print(len(np.unique(genre)))
num_clusters = 5

km = KMeans(n_clusters=num_clusters)

km.fit(tfidf_matrix)

clusters = km.labels_.tolist()
import joblib

joblib.dump(km, 'doc_cluster.pkl')
# comment once the model is run and saved

km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()
films = {'titles': title, 'plot': plot, 'cluster': clusters, 'genre': genre}
# print(films)
frame = pd.DataFrame(films, columns=['titles', 'cluster', 'genre'])
print(frame)

frame['cluster'].value_counts()

print("Top terms per cluster:")
print()
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')

    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print() #add whitespace
    print() #add whitespace

    print("Cluster %d titles:" % i, end='')
    for title in frame.ix[i]['titles'].values.tolist():
        print(' %s,' % title, end='')
    print() #add whitespace
    print() #add whitespace

print()
print()