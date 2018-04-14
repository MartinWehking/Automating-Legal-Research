import tfidf_model as tf

import pandas as pd
from os import getcwd, path
import csv
import random
import logging


def create_data():
    df = pd.read_csv('datasets/dataset3.csv', encoding='utf-8', sep="|")
    df = df.groupby(['section', 'title'])['paragraph'].apply(lambda x: ' '.join(map(str, x))).reset_index()
    data = tf.preprocess(df)
    return data


def add_result(books, score, file=getcwd() + '/results2.csv'):
    exists = True
    if not path.isfile(file):
        exists = False
    with open(file, 'a+') as csv_file:
        header = ['books', 'score']
        writer = csv.DictWriter(csv_file, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow({'books': ','.join(books), 'score': score})


def evaluate(data, combs):
    for comb in combs:
        sub_data = None
        for book in comb:
            if sub_data is not None:
                sub_data = sub_data.append(data.loc[(data.section.str.startswith(book + ' '))])
            else:
                sub_data = data.loc[(data.section.str.startswith(book + ' '))]
        vec, matrix = tf.vectorize(sub_data)
        _, score = tf.cluster(matrix, get_book_label(sub_data))
        add_result(comb, score)


def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)


def count_books(df):
    df['n'] = df['books'].apply(lambda x: x.count(',') + 1)
    return df


def best_results(df):
    max_results = df.groupby(['n'])['score'].max().tolist()
    best = {}
    for score in max_results:
        best[df.loc[df['score'] == score]['books'].tolist()[0]] = score
    return best


def worst_results(df):
    max_results = df.groupby(['n'])['score'].min().tolist()
    best = {}
    for score in max_results:
        best[df.loc[df['score'] == score]['books'].tolist()[0]] = score
    return best


def get_book_label(df):
    titles = df['section'].apply(lambda x: x.split(' ')[0])
    return titles.tolist()


def get_results(result_file):
    results = pd.read_csv(result_file)
    results = count_books(results)
    max_results = best_results(results)
    logging.info('Best results: '+ str(max_results))
    min_results = worst_results(results)
    logging.info('Worst results: ' + str(min_results))


if __name__ == '__main__':
    d = create_data()
    books = d['section'].apply(lambda x: x.split(' ')[0])
    books = set(books.tolist())
    for i in range(30):
        i = i + 2
        c = [random_combination(books, i) for _ in range(200)]
        c = [comb for comb in c if len(set(comb)) > 1]
        evaluate(d, c)
