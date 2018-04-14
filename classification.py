import numpy as np
from sklearn.svm import LinearSVC
from sklearn import metrics
import logging


def classify_embeddings(data, label, training_percentage=0.8, repetitions=1, random_seed=100):
    cut = int(len(data) * training_percentage)
    macro_score = micro_score = acc = 0
    indices = [i for i in range(len(data))]
    for index in range(repetitions):
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        test_data = np.array([data[x] for x in indices[cut:]])
        train_data = np.array([data[x] for x in indices[:cut]])
        test_label = np.array([label[x] for x in indices[cut:]])
        train_label = np.array([label[x] for x in indices[:cut]])
        shape_1, shape_2 = np.shape(test_data)[0], np.shape(test_data)[2]
        test_data = np.reshape(test_data, (shape_1, shape_2))
        shape_1, shape_2 = np.shape(train_data)[0], np.shape(train_data)[2]
        train_data = np.reshape(train_data, (shape_1, shape_2))
        sl = LinearSVC()
        sl.fit(train_data, train_label)
        predicted = sl.predict(test_data)
        acc += metrics.accuracy_score(test_label, predicted)
        micro_score += metrics.f1_score(test_label, predicted, average='micro')
        macro_score += metrics.f1_score(test_label, predicted, average='macro')
    logging.info("Accuracy: %s", acc)
    logging.info("Micro F1: %s", micro_score)
    logging.info('Macro F1: %s', macro_score)

