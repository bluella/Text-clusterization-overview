#!/usr/bin/env python3
"""Module with all data manipulations
   P.S. Can be converted to .ipynb"""
# %%
# All the imports
import re
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# from sklearn.feature_extraction.text import CountVectorizer  # For Bag of words
from sklearn.feature_extraction.text import TfidfVectorizer  # For TF-IDF
from sklearn.cluster import KMeans

import gensim  # For Word2Vec

import matplotlib.pyplot as plt

import tco.helpers as hf


# %%
# Load dataset to work
df_posts = hf.read_posts_csv(hf.datasets_dir +
                             'posts_Reddit_wot_en.csv')

# %%
# Get dataset overview
hf.print_df_info(df_posts)

# %%
# Basic cleaning of dataset
df_posts.drop_duplicates(inplace=True)
df_posts['text'] = df_posts['text'].astype('str')

# %%
# create X
X = df_posts['text']

# %%
# tokenization, lemmatizing and cleaning
stop = set(stopwords.words('english'))
temp = []
lemmatizer = WordNetLemmatizer()
for sentence in X:
    sentence = sentence.lower()                 # Converting to lowercase
    html_tags = re.compile('<.*?>')
    sentence = re.sub(r'http\S+', r'', sentence)  # removing links
    sentence = re.sub(html_tags, ' ', sentence)  # Removing HTML tags
    # Removing Punctuations
    sentence = sentence.translate(str.maketrans("", "", string.punctuation))

    lem_words = [lemmatizer.lemmatize(
        word) for word in word_tokenize(sentence)]  # lemmatizing
    words = [word for word in lem_words if (
        word not in stop and len(word) > 2)]    # stopword removal

    temp.append(words)

tokenized_X = temp

# assemble sentences back from words
sent = []
for row in tokenized_X:
    sequ = ''
    for word in row:
        sequ = sequ + ' ' + word
    sent.append(sequ)

texts_X = sent
print(texts_X[:10])

# %%
# declare variables for future purposes
true_k = 5
cluster_colors = {0: '#1b9e77', 1: '#d95f02',
                  2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

# %%
# tfdf vectorization

# count_vectorizer = CountVectorizer() # uncomment in case of change TfidfVectorizer
tfdf_vectorizer = TfidfVectorizer(
    max_df=0.95, min_df=0.001, max_features=10**5, ngram_range=(1, 3))
tfdf_data = tfdf_vectorizer.fit_transform(texts_X)
print(tfdf_data[1])
# always check to make sure words were converted to vec right
print(tfdf_data.shape)

# %%
# prepare sentences for doc2vec
labeled_sentence = gensim.models.doc2vec.TaggedDocument
all_content_train = []
j = 0
for em in tokenized_X:
    all_content_train.append(labeled_sentence(em, [j]))
    j += 1

# %%
# define and train doc2vec model
d2v_model = gensim.models.Doc2Vec(all_content_train,
                                  vector_size=100,
                                  window=2,
                                  min_count=50,
                                  workers=7,
                                  dm=0 # 0 for PV-DBOW 1 fof PV-DM
                                  )
d2v_model.train(all_content_train,
                total_examples=d2v_model.corpus_count,
                epochs=d2v_model.epochs
                )
# %%
# creating KMeans for tfdf_data
kmeans_tfdf = KMeans(n_clusters=true_k, init='k-means++',
                     max_iter=100, n_init=5)
kmeans_tfdf.fit(tfdf_data)
labels_tfdf = kmeans_tfdf.labels_.tolist()

# lets get a view on created clusters
df_posts['label_tfdf'] = labels_tfdf
for cl_n in range(true_k):
    hf.print_cluster_posts(df_posts, 'label_tfdf', cluster=cl_n,
                           dirname=hf.datasets_dir + 'posts_tfdf/',
                           filename='{}_posts.txt'.format(cl_n))
print(df_posts['label_tfdf'].value_counts())

# %%
# creating KMeans for d2v_data
kmeans_d2v = KMeans(n_clusters=true_k, init='k-means++', max_iter=100)
d2v_data = kmeans_d2v.fit(d2v_model.docvecs.doctag_syn0)
labels_d2v = kmeans_d2v.labels_.tolist()

# lets get a view on created clusters
df_posts['label_d2v'] = labels_d2v
for cl_n in range(true_k):
    hf.print_cluster_posts(df_posts, 'label_d2v', cluster=cl_n,
                           dirname=hf.datasets_dir + 'posts_d2v/',
                           filename='{}_posts.txt'.format(cl_n))
print(df_posts['label_d2v'].value_counts())

# %%
# get most popular terms for tfdf_model
print("Top terms per cluster:")
order_centroids = kmeans_tfdf.cluster_centers_.argsort()[:, ::-1]
terms = tfdf_vectorizer.get_feature_names()
cluster_names = {}
for i in range(true_k):
    print("Cluster %d:" % i)
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])
    cluster_names[i] = terms[order_centroids[i, 0]] + \
        ',' + terms[order_centroids[i, 1]]

# %%
# tfdf visualize
dense_matrix = tfdf_data.todense()
hf.plot_pca(dense_matrix,
            kmeans_tfdf.cluster_centers_,
            labels_tfdf,
            cluster_colors)

hf.plot_mds(tfdf_data,
            labels_tfdf,
            cluster_colors)
plt.show()


# %%
# d2v visualize
hf.plot_pca(d2v_model.docvecs.doctag_syn0,
            kmeans_d2v.cluster_centers_,
            labels_d2v,
            cluster_colors)

hf.plot_mds(d2v_model.docvecs.doctag_syn0,
            labels_d2v,
            cluster_colors)
plt.show()

#%%
print(type(d2v_model.docvecs.doctag_syn0))
print(type(dense_matrix))


#%%
