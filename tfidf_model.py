import re
import nltk
from nltk.corpus import stopwords
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import pandas as pd
from time import time
from sklearn.decomposition import PCA, TruncatedSVD
from matplotlib import pyplot as plt
import logging
import numpy as np
import sys


# Expects a dataframe that contains a column 'paragraph' and returns a dataframe
# without stopwords and with added columns
# 'tokenized_pars', 'lemmatized_toks' and 'part_of_speech'
def preprocess(df):
    logging.info("Preprocess:")
    t0 = time()

    logging.info("Preprocess:\t Remove stopwords, special characters and make lower case")
    stops = frozenset(stopwords.words('english')) | frozenset(("petitioner", "complainant", "plaintiff", "plaintiff-respondent", "court"))
    df['paragraph'] = df['paragraph'].apply(str)
    df['paragraph'] = df['paragraph'].str.lower()
    df['paragraph'] = df['paragraph'].str.replace('[\(\[](repealed|no longer applicable|omitted)[\]\)]', '')  # remove repealed paragraphs
    df['paragraph'] = df['paragraph'].str.replace('[^a-z%§ -]', '')  # remove uninteresting special characters
    pattern = re.compile("|".join(map(lambda x: "(?<![a-z-])" + x + "(?![a-z-])", stops)))
    df['paragraph'] = df['paragraph'].map(lambda x: pattern.sub("", x))  # remove stopwords
    pattern = re.compile(" +")
    df['paragraph'] = df['paragraph'].map(lambda x: pattern.sub(" ", x))  # remove multiple consecutive spaces

    df.drop(df[df['paragraph'] == ""].index, inplace=True)  # drop empty paragraphs

    logging.info("Preprocess:\t Tokenize paragraphs")
    df['tokenized_pars'] = df.apply(lambda row: nltk.word_tokenize(row['paragraph']), axis=1)

    logging.info("Preprocess:\t Lemmatize tokens")
    df['lemmatized_toks'] = df['tokenized_pars'].apply(
        lambda row: [nltk.stem.WordNetLemmatizer().lemmatize(tok) for tok in row])

    # part-of-speech-tagging
    df['part_of_speech'] = df['lemmatized_toks'].apply(lambda row: [nltk.pos_tag(row)])

    logging.info("Preprocess:\t Join lemmas")
    df['lemmatized_sents'] = df['lemmatized_toks'].apply(lambda row: ' '.join(item for item in row))

    def jnnnTerms(x):
        terms = []
        i = 0
        while i < len(x) - 1:
            if (x[i][1].startswith('NN') or x[i][1].startswith('JJ')) and x[i+1][1].startswith('NN'):
                terms.append(x[i][0] + " " + x[i + 1][0])
            i += 1
        return terms


    df['terms'] = df['part_of_speech'].map(lambda x: " ".join([z for y in x for z in jnnnTerms(y)]))

    logging.info("Preprocess:\t Done in %0.3fs" % (time() - t0))

    return df


# Clustering
def cluster(sparse_matrix, labels, algorithm='kmeans'):
    logging.info("Cluster:")

    true_k = np.unique(labels).shape[0]
    t0 = time()
    if algorithm == 'kmeans':
      km = KMeans(n_clusters=true_k)
    else:
      km = AgglomerativeClustering(n_clusters=true_k, linkage='average', affinity='cosine')
      sparse_matrix = [x if len([y for y in x if y != 0]) > 0 else list(map(lambda y: sys.float_info.epsilon, x)) for x in sparse_matrix.toarray().tolist()]  # todense()

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

    clf = PCA(n_components=2)
    pca_embeddings = clf.fit_transform(sparse_matrix.toarray())

    color_map = [int(10 * c_label[index]) for index in range(len(pca_embeddings))]

    plt.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], c=color_map, alpha=0.5)
    plt.show()

    logging.info("Visualize:\t Done in %0.3fs", (time() - t0))


# Expects a text which should be searched in the given matrix by using the trained vectorizer
def search(text, matrix, vectorizer):
    logging.info("Search:\t")
    t0 = time()

    to_search = pd.DataFrame(
        {'paragraph': [text]})

    to_search = preprocess(to_search)
    to_search_matrix = vectorizer.transform(to_search['lemmatized_sents'].tolist())

    cosine_similarities = metrics.pairwise.cosine_similarity(to_search_matrix, matrix).flatten()
    n = list(map(lambda x: x[0], sorted(enumerate(cosine_similarities), key=lambda x: x[1], reverse=True)))[:5]

    logging.info("Search:\t Done in %0.3fs", (time() - t0))

    return n


# Expects a dataframe which should be vectorized
# Returns the TfIdfVectorizer and the calculated Matrix
def vectorize(df):
    logging.info("Vectorize:")
    t0 = time()

    corpus = df['lemmatized_sents'].tolist()
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform(raw_documents=corpus)

    logging.info("Vectorize:\t Done in %0.3fs", (time() - t0))

    return tfidf_vectorizer, tfidf_matrix
