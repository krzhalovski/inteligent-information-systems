import numpy as np
import pandas as pd
import networkx as nx

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn import feature_extraction, model_selection

from katz import katz
from random_walk import random_walk

import matplotlib
matplotlib.use('agg') # za da raboti so python3.7
import matplotlib.pyplot as plt


from katz import katz
from random_walk import random_walk


def read_graph():
    # Load the graph from edgelist
    edgelist = pd.read_table("data/cora.cites",
                             header=None, names=["source", "target"])
    edgelist["label"] = "cites"
    graph = nx.from_pandas_edgelist(edgelist, edge_attr="label")
    nx.set_node_attributes(graph, "paper", "label")

    # Load the features and subject for the nodes
    feature_names = ["w_{}".format(ii) for ii in range(1433)]
    column_names = feature_names + ["subject"]
    node_data = pd.read_table("data/cora.content",
                              header=None, names=column_names)

    return graph, node_data, feature_names


def split_data(node_data):
    train_data, test_data = model_selection.train_test_split(node_data, train_size=0.7, test_size=None,
                                                             stratify=node_data['subject'])
    return train_data, test_data


def encode_classes(train_data, test_data):
    target_encoding = feature_extraction.DictVectorizer(sparse=False)

    train_targets = target_encoding.fit_transform(train_data[["subject"]].to_dict('records'))
    test_targets = target_encoding.transform(test_data[["subject"]].to_dict('records'))

    return train_targets, test_targets


def calculate_neighbors_class_features(graph, train_nodes, test_nodes, train_targets, test_targets):
    adjacency = nx.adjacency_matrix(graph).toarray()
    np.fill_diagonal(adjacency, val=0.0)
    indices = {x: i for i, x in enumerate(list(graph.nodes()))}
    train_indices = [indices[node] for node in train_nodes]
    test_indices = [indices[node] for node in test_nodes]

    train_new_features = np.zeros(train_targets.shape)
    for i, node in enumerate(train_indices):
        values = adjacency[node]
        predicted_probs = np.sum(train_targets.T * values[train_indices], axis=1)
        train_new_features[i, :] = predicted_probs

    test_new_features = np.zeros(test_targets.shape)
    for i, node in enumerate(test_indices):
        values = adjacency[node]
        predicted_probs = np.sum(train_targets.T * values[train_indices], axis=1)
        test_new_features[i, :] = predicted_probs

    return train_new_features, test_new_features


def calculate_random_walk_features(graph, train_nodes, test_nodes, train_targets, test_targets):
    adjacency = nx.adjacency_matrix(graph, nodelist=graph.nodes(), weight=None).toarray()
    indices = {x: i for i, x in enumerate(list(g.nodes()))}
    train_indices = [indices[name] for name in train_nodes]
    test_indices = [indices[name] for name in test_nodes]

    train_targets_7xM = train_targets.reshape(7, -1)
    train_new_features = np.zeros(train_targets.shape)
    for i, train_node in enumerate(train_indices):
        probabilities = random_walk(adjacency, train_node)
        probabilities_Mx1 = probabilities[train_indices].reshape(-1, 1)
        probabilities_Mx1[train_node] = 0
        multiplication = np.matmul(train_targets_7xM, probabilities_Mx1)
        train_new_features[i, :] = multiplication.ravel()

    test_new_features = np.zeros(test_targets.shape)
    for i, test_node in enumerate(test_indices):
        probabilities = random_walk(adjacency, test_node)
        probabilities_Mx1 = probabilities[train_indices].reshape(-1, 1)
        multiplication = np.matmul(train_targets_7xM, probabilities_Mx1)
        train_new_features[i, :] = multiplication.ravel()

    return train_new_features, test_new_features


def calculate_katz_features(graph, train_nodes, test_nodes, train_targets, test_targets):
    matrix_of_similarity, nodelist= katz(graph)

    adjacency = nx.adjacency_matrix(graph).toarray()
    np.fill_diagonal(adjacency, val=0.0)
    indices = {x: i for i, x in enumerate(list(graph.nodes()))}
    train_indices = [indices[node] for node in train_nodes]
    test_indices = [indices[node] for node in test_nodes]

    train_targets_7xM = train_targets.reshape(7, -1)
    train_new_features = np.zeros(train_targets.shape)

    for i, train_node in enumerate(train_indices):
        values = matrix_of_similarity[train_node]
        probabilities_Mx1 = values[train_indices].reshape(-1, 1)
        probabilities_Mx1[train_node] = 0
        multiplication = np.matmul(train_targets_7xM, probabilities_Mx1)
        train_new_features[i, :] = multiplication.ravel()

    test_new_features = np.zeros(test_targets.shape)
    for i, test_node in enumerate(test_indices):
        values = matrix_of_similarity[test_node]
        probabilities_Mx1 = values[train_indices].reshape(-1, 1)
        multiplication = np.matmul(train_targets_7xM, probabilities_Mx1)
        test_new_features[i, :] = multiplication.ravel()

    return train_new_features, test_new_features

def create_features(graph, train_nodes, test_nodes, train_targets, test_targets):
    # First order neighbors classes
    train_n, test_n = calculate_neighbors_class_features(graph, train_nodes, test_nodes,
                                                         train_targets, test_targets)
    # Random Walk
    train_pr, test_pr = calculate_random_walk_features(graph, train_nodes, test_nodes, train_targets, test_targets)
    # Katz
    train_k, test_k = calculate_katz_features(graph, train_nodes, test_nodes, train_targets, test_targets)

    return np.hstack((train_n, train_pr, train_k)), np.hstack((test_n, test_pr, test_k))


def calculate_metrics(test_targets, predictions):
    """Calculation of accuracy score, F1 micro and F1 macro"""
    print(f'\tAccuracy score: {accuracy_score(test_targets, predictions)}')
    print(f'\tF1-micro: {f1_score(test_targets, predictions, average="micro")}')
    print(f'\tF1-macro: {f1_score(test_targets, predictions, average="macro")}')


def classification_by_title(train_features, test_features, train_targets, test_targets):
    """Classification using only the words in the title"""
    classifier = RandomForestClassifier(n_estimators=100, random_state=0)
    classifier.fit(train_features, train_targets)
    predictions = classifier.predict(test_features)
    calculate_metrics(test_targets, predictions)


def classification_by_random_walk(graph, train_nodes, test_nodes, train_targets, test_targets):
    """Classification using only the values from Random Walk"""
    adj = nx.adjacency_matrix(graph, nodelist=graph.nodes(), weight=None).toarray()
    indices = {x: i for i, x in enumerate(list(graph.nodes()))}

    train_ind = [indices[node] for node in train_nodes]
    test_int = [indices[node] for node in test_nodes]
    predictions = list()

    tt_reshaped = train_targets.reshape(7, -1)
    for indice, node in zip(range(813), test_nodes):
        prediction = random_walk(adj, indice)
        reshaped = prediction[train_ind].reshape(-1, 1)
        multiplied = np.matmul(tt_reshaped, reshaped)
        predicted = np.argmax(multiplied)
        formated = np.zeros(7)
        formated[predicted] = 1
        predictions.append(formated)

    predictions = np.array(predictions)
    calculate_metrics(test_targets, predictions)
    return
    pass


def classification_by_combined_features(graph, train_nodes, test_nodes, train_targets, test_targets):
    """Classification using combined features from the structure of the graph"""
    train_new_features, test_new_features = create_features(graph, train_nodes,
                                                            test_nodes, train_targets,
                                                            test_targets)
    pass


def main1():
    g, nodes, features_names = read_graph()

    degrees = [x[1] for x in list(nx.degree(g))]
    plt.hist(degrees)

    cc = nx.connected_components(g)
    #comms = nx.algorithms.community.girvan_newman(g)

    print([x for x in cc])
    return


def main2():
    g, nodes, features_names = read_graph()
    train_data, test_data = split_data(nodes)
    train_targets, test_targets = encode_classes(train_data, test_data)
    train_features, test_features = train_data[features_names], test_data[features_names]
    train_nodes = train_features.index.values.tolist()
    test_nodes = test_features.index.values.tolist()

    print('Classification by title:')
    classification_by_title(train_features, test_features, train_targets, test_targets)

    print('Classification by random walks:')
    #classification_by_random_walk(g, train_nodes, test_nodes, train_targets, test_targets)

    print('Classification by combined features:')
    classification_by_combined_features(g, train_nodes, test_nodes,
                                        train_targets, test_targets)

    return


if __name__ == '__main__':
    main2()

