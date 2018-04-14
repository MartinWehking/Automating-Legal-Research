import logging

import nltk
import re
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.corpus import stopwords
import numpy as np
from time import time
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import pandas as pd


def create_d2v_embeddings(data, pre_pro='lemmatized_toks', key='section'):
    logging.info("Create embeddings:")
    t0 = time()

    tagged_docs = data.apply(lambda x: TaggedDocument(x[pre_pro], [x[key]]), axis=1)
    d2v = Doc2Vec(tagged_docs.tolist(), negative=20, vector_size=64, epochs=30, dm=0, alpha=0.05, hs=1,
                  min_alpha=0.01)
    doc_vec = [d2v.docvecs[k] for k in data[key]]
    logging.info("Create embeddings:\t Done in %0.3fs", (time() - t0))
    return d2v, np.asmatrix(doc_vec)


def cluster(sparse_matrix, labels, algorithm='kmeans'):
    logging.info("Cluster:")

    true_k = np.unique(labels).shape[0]
    t0 = time()
    if algorithm == 'kmeans':
      km = KMeans(n_clusters=true_k)
    else:
      km = AgglomerativeClustering(n_clusters=true_k, linkage='average', affinity='cosine')
    labels_ = km.fit_predict(sparse_matrix)

    logging.info("Cluster:\t Homogeneity: %0.3f", metrics.homogeneity_score(labels, labels_))
    logging.info("Cluster:\t Completeness: %0.3f", metrics.completeness_score(labels, labels_))
    logging.info("Cluster:\t V-measure: %0.3f", metrics.v_measure_score(labels, labels_))
    logging.info("Cluster:\t Adjusted Rand-Index: %.3f"
                 , metrics.adjusted_rand_score(labels, labels_))
    logging.info("Cluster:\t Silhouette Coefficient: %0.3f"
                 , metrics.silhouette_score(sparse_matrix, labels_, sample_size=1000))

    logging.info("Cluster:\t Done in %0.3fs", (time() - t0))

    return labels_, metrics.v_measure_score(labels, labels_)


def visualize_cluster(sparse_matrix, c_label):
    logging.info("Visualize:")
    t0 = time()

    pca_embeddings = PCA(n_components=2).fit_transform(sparse_matrix)
    color_map = [int(10 * c_label[index]) for index in range(len(pca_embeddings))]

    plt.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], c=color_map, alpha=0.5)
    plt.show()

    logging.info("Visualize:\t Done in %0.3fs", (time() - t0))

def search(text, matrix, d2v):
    logging.info("Search:\t")
    t0 = time()

    to_search = pd.DataFrame(
        {'paragraph': [text]})

    to_search = preprocess(to_search)
    to_search = d2v.infer_vector(to_search['lemmatized_sents'].tolist())

    to_search = to_search.reshape(1, -1)

    cosine_similarities = metrics.pairwise.cosine_similarity(to_search, matrix).flatten()
    n = list(map(lambda x: x[0], sorted(enumerate(cosine_similarities), key=lambda x: x[1], reverse=True)))[:5]

    logging.info("Search:\t Done in %0.3fs", (time() - t0))

    return n


# Expects a dataframe that contains a column 'paragraph' and returns a dataframe
# without stopwords and with added columns
# 'tokenized_pars', 'lemmatized_toks' and 'part_of_speech'
def preprocess(df):
    logging.info("Preprocess:")
    t0 = time()

    logging.info("Preprocess:\t Remove stopwords, special characters and make lower case")
    stops = frozenset(stopwords.words('english'))
    df['paragraph'] = df['paragraph'].apply(str)
    df['paragraph'] = df['paragraph'].str.lower()
    df['paragraph'] = df['paragraph'].str.replace('[\(\[](repealed|no longer applicable|omitted)[\]\)]', '')  # remove repealed paragraphs
    df['paragraph'] = df['paragraph'].str.replace('[^a-z%§ ]', '')  # remove uninteresting special characters
    pattern = re.compile("|".join(map(lambda x: "(?<![a-z])" + x + "(?![a-z])", stops)))
    df['paragraph'] = df['paragraph'].map(lambda x: pattern.sub("", x))  # remove stopwords
    pattern = re.compile(" +")
    df['paragraph'] = df['paragraph'].map(lambda x: pattern.sub(" ", x))  # remove multiple consecutive spaces

    df.drop(df[df['paragraph'] == ""].index, inplace=True)  # drop empty paragraphs

    logging.info("Preprocess:\t Tokenize paragraphs")
    df['tokenized_pars'] = df.apply(lambda row: nltk.word_tokenize(row['paragraph']), axis=1)

    logging.info("Preprocess:\t Lemmatize tokens")
    df['lemmatized_toks'] = df['tokenized_pars'].apply(
        lambda row: [nltk.stem.WordNetLemmatizer().lemmatize(tok) for tok in row])

    logging.info("Preprocess:\t Join lemmas")
    df['lemmatized_sents'] = df['lemmatized_toks'].apply(lambda row: ' '.join(item for item in row))

    logging.info("Preprocess:\t Done in %0.3fs" % (time() - t0))

    return df
