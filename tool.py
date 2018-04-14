import pandas as pd
import json
import doc2vec_model
import scrapper
import sys
import logging

from collections import Counter
from optparse import OptionParser
from classification import classify_embeddings
from evaluation import get_results
import matplotlib.pyplot as plt

import tfidf_model


def get_book_label(df):
    titles = df['section'].apply(lambda x: x.split(' ')[0])
    return titles.tolist()


def get_division_label(df):
    titles = df['division'].apply(lambda x: x.split(':'))
    return [label[1] if label[1] != '' else 'Dummy Division' for label in titles]


# Expects a list with urls to scrape
def get_datafrome_from_urls(url_list):
    texts = scrapper.getTexts(url_list)
    columns = ['section', 'title', 'division', 'paragraph']

    dataframe = pd.DataFrame(texts, columns=columns)
    dataframe = dataframe.groupby(['section', 'title', 'division'])['paragraph'].apply(
        lambda x: ' '.join(map(str, x))).reset_index()

    return dataframe


def unique_words(df):
    df = tfidf_model.preprocess(df)
    freq_all = Counter(" ".join(df["paragraph"]).split())
    for book in set(get_book_label(df)):
        freq = Counter(" ".join(df.loc[(df.section.str.startswith(book + ' '))]["paragraph"]).split())
        unique = [word for word in freq.keys() if freq[word] / freq_all[word] == 1.]
        logging.info(book + ': ' + str(len(unique)))


def count_paragraphs(df):
    df = tfidf_model.preprocess(df)
    df['section'] = df['section'].apply(lambda x: x.split()[0])
    count = df.groupby('section')['title'].count()
    count = count.to_dict()
    logging.info(count)
    return count


def create_pie_chart(df):
    count = count_paragraphs(df)
    count = sorted(count.items(), reverse=True, key=lambda x: x[1])
    labels = list(map(lambda x: x[0], count))
    sizes = list(map(lambda x: x[1], count))

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes,
           shadow=True, startangle=90, labels=labels)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()


def most_comm(df, n):
    df = tfidf_model.preprocess(df)
    for book in set(get_book_label(df)):
        logging.info(
            Counter(" ".join(df.loc[(df.section.str.startswith(book + ' '))]["paragraph"]).split()).most_common(n))


def execute_tfidf(df, cluster=False, visualize=False, search_text="", classify=False, division_labels="book",
                  algorithm='kmeans'):
    if df is None:
        logging.error("No data set provided. Exiting!")
        sys.exit(1)

    model = tfidf_model.preprocess(df)
    vectorizer, matrix = tfidf_model.vectorize(model)

    if cluster:
        if division_labels == "book":
            label, _ = tfidf_model.cluster(matrix, get_book_label(model), algorithm=algorithm)
        else:
            label, _ = tfidf_model.cluster(matrix, get_division_label(model), algorithm=algorithm)

        if visualize:
            tfidf_model.visualize_cluster(matrix, label)

    if search_text:
        idx = tfidf_model.search(search_text, matrix, vectorizer)
        for i in idx:
            logging.info("Result: %s", df.section[i])

    if classify:
        classify_embeddings(matrix.todense(), get_book_label(model))


def execute_doc2vec(df, cluster=False, visualize=False, search_text="", classify=False, division_labels="book",
                    algorithm='kmeans'):
    if df is None:
        logging.error("No data set provided. Exiting!")
        sys.exit(1)

    model = doc2vec_model.preprocess(df)
    d2v, matrix = doc2vec_model.create_d2v_embeddings(model)

    if cluster:
        if division_labels == "book":
            label, _ = doc2vec_model.cluster(matrix, get_book_label(model), algorithm=algorithm)
        else:
            label, _ = doc2vec_model.cluster(matrix, get_division_label(model), algorithm=algorithm)
        if visualize:
            doc2vec_model.visualize_cluster(matrix, label)

    if search_text:
        idx = doc2vec_model.search(search_text, matrix, d2v)
        for i in idx:
            logging.info("Result: %s", df.section[i])

    if classify:
        classify_embeddings(matrix, get_book_label(model))


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    op = OptionParser()
    op.add_option('--classify',
                  dest='classify', action='store_true',
                  help='Classify documents with svm. Only works with tfidf and/or doc2vec option')
    op.add_option("--cluster",
                  dest="cluster", action="store_true",
                  help="Cluster documents with k-means. Only works with tfidf and/or doc2vec option")
    op.add_option('--algorithm',
                  dest='algorithm', type='string',
                  help='Defines the clustering algorithm. Only hac and kmeans are possible values.')
    op.add_option("--config",
                  dest="config", type="string",
                  help="Load configuration from file. Config file always has top priority above other settings. "
                       "Currently supports a list of URLs and what to choose as a cluster label type "
                       "(book or division title)")
    op.add_option("--dataset",
                  dest="dataset", type="string",
                  help="Load pandas data set from file. Does not work with url option. Url is always evaluated first")
    op.add_option("--doc2vec",
                  dest="doc2vec", action="store_true",
                  help="Calculate word embeddings with doc2vec")
    op.add_option("--out",
                  dest="out", type="string",
                  help="Set path to outfile for the dataset")
    op.add_option("--search",
                  dest="search", type="string",
                  help="Search similiar text in the dataset")
    op.add_option("--tfidf",
                  dest="tfidf", action="store_true",
                  help="Calculate a feature matrix using tfidf")
    op.add_option("--visualize",
                  dest='visualize', action='store_true',
                  help='Visualize the embeddings. Only works with cluster option')
    op.add_option("--url",
                  dest="url", type="string",
                  help="Scrap webpage of the given url to retrieve law code text. Only english texts at "
                       "'gesetze-im-internet.de supported atm.")
    op.add_option('--results',
                  dest='results', type='string',
                  help='Return information about the clustering evaluation')
    op.add_option('--freqs',
                  dest='freqs', type='int',
                  help='Return the n most frequent words of each book.')
    op.add_option('--unique',
                  dest='unique', action='store_true',
                  help='Count the number of unique words per book.')
    op.add_option('--count',
                  dest='count', action='store_true',
                  help='Count the paragraphs of each book.')

    (opts, args) = op.parse_args(sys.argv[1:])
    df = None

    labeltype = "book"
    if opts.config:
        config = json.load(open(opts.config))
        df = get_datafrome_from_urls(config['urls'])
        if config['labeltype'] is not None:
            labeltype = config['labeltype']

    if opts.url and df is None:
        df = get_datafrome_from_urls([opts.url])

    if opts.dataset is not None and df is None:
        df = pd.read_csv(opts.dataset, encoding='utf-8', sep="|")
        df = df.groupby(['section', 'title'])['paragraph'].apply(
            lambda x: ' '.join(map(str, x))).reset_index()

    if opts.tfidf:
        execute_tfidf(df, opts.cluster, opts.visualize, opts.search, opts.classify, labeltype, opts.algorithm)

    if opts.doc2vec:
        execute_doc2vec(df, opts.cluster, opts.visualize, opts.search, opts.classify, labeltype, opts.algorithm)

    if opts.out:
        df.to_csv(opts.out, encoding='utf-8', sep="|")

    if opts.results:
        get_results(opts.results)

    if opts.freqs:
        most_comm(df, opts.freqs)

    if opts.unique:
       unique_words(df)

    if opts.count:
        count_paragraphs(df)
