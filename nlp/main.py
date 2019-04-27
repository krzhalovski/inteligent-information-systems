import numpy as np
import pandas as pd
import argparse as ap
import nltk
import json
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics


def init_parser():
    parser = ap.ArgumentParser()
    parser.add_argument("-p", "--part",
                        help="Part of assignment to solve [1 or 2]",
                        type=int,
                        default=1)
    return parser


def format_data(data):

    for review in data:
        if review['Stars'] < 4:
            review['Stars'] = 0
        else:
            review['Stars'] = 1

    return data


def create_features(data):
    print('Generating features')

    words = dict()
    i = 0
    vecs = [0] * 31838
    for review in data:
        line = review['Text'].split(' ')
        for word in line:
            if word in words:
                continue
            else:
                words[word] = 1
                vecs[i] = word
                i += 1

    return vecs


def map_reviews(data, features):
    mapped = []
    for review in data:
        words = set(review['Text'].split(' '))
        feat = [0] * 31838
        i = 0
        for feature in features:
            if feature in words:
                feat[i] = 1
            i += 1
        mapped.append(feat)
    return mapped


def get_targets(data):
    targets = [0] * len(data)
    i=0
    for review in data:
        targets[i] = review['Stars']
        i += 1
    return targets


def print_accuracy(targets, predictions):
    print("Accuracy is: ", metrics.accuracy_score(targets, predictions))
    print("Precision is: ", metrics.precision_score(targets, predictions, average='micro'))
    print("Recall is: ", metrics.recall_score(targets, predictions, average='micro'))
    print("F1-macro is: ", metrics.f1_score(targets, predictions, average='macro'))
    return


def random_forest(data):
    np.random.shuffle(data)
    train = data[:2000]
    test = data[2000:]

    features = create_features(data)
    train_features = map_reviews(train, features)
    test_features = map_reviews(test, features)
    train_targets = get_targets(train)
    test_targets = get_targets(test)

    classifier = RandomForestClassifier(n_estimators=100, random_state=0)
    classifier.fit(train_features, train_targets)
    predictions = classifier.predict(test_features)
    print_accuracy(test_targets, predictions)
    return


def main1():
    with open('data/yelp_reviews_subset.json', 'r', encoding='utf-8') as doc:
        data = json.load(doc)

    random_forest(data)
    return

def gt(data):
    targets = [0] * len(data)
    i = 0
    for review in data:
        targets[i] = review['']
        i += 1
    return targets

def cf(data):
    return


def rf(data):
    train = data[:4000]
    test = data[4000:]

    train_features = 1
    train_targets = 1

    return


def main2():

    data = pd.read_table('data/SemEval2016Subset.csv', sep='###', header=None)

    rf(data)
    return


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()

    if args.part == 1:
        main1()
    else:
        main2()
