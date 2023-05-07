import logging
import sys

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s : [%(levelname)s]  %(message)s',
    handlers=[
        logging.FileHandler('supervised.log'),
        logging.StreamHandler(sys.stdout)
    ]
)


import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, HinSAGE, link_classification
from stellargraph import globalvar
from stellargraph import datasets
from stellargraph.ensemble import BaggingEnsemble

import tensorflow as tf
from tensorflow import keras
import random as python_random
import warnings
warnings.filterwarnings("ignore")

from sklearn import preprocessing, feature_extraction, model_selection
import os
import numpy as np
import pandas as pd
from timeit import default_timer as timer

tf.random.set_seed(42)
np.random.seed(42)
python_random.seed(42)

class Model:

    def __init__(self,nodes,edges):
        self.model = None

        self.nodes =  nodes
        self.edges = edges

        self.graph_train = None
        self.graph_val = None
        self.graph_test = None

        self.train_gen = None

        self.train_flow = None
        self.val_flow = None
        self.test_flow = None

        self.edge_ids_train = None
        self.edge_labels_train = None

        self.n_estimators = 2

    def initialize(self, **hyper_params):

        if(not "batch_size" in hyper_params.keys()):
            batch_size = 40
        if(not "layer_sizes" in hyper_params.keys()):
            num_samples = [20, 10]
        if(not "num_samples" in hyper_params.keys()):
            layer_sizes = [15, 15]
        if(not "bias" in hyper_params.keys()):
            bias = True
        if(not "dropout" in hyper_params.keys()):
            dropout = 0.1
        if(not "lr" in hyper_params.keys()):
            lr = 1e-2
        # previously 3, 6
        self.n_estimators = 2  # Number of models in the ensemble
        n_predictions = 6  # Number of predictions per query point per model

        test_edges = self.edges.iloc[int(self.edges.shape[0] * 0.6):]
        if 'weight' in test_edges.columns:
            test_edges = test_edges.drop(['weight'], axis=1)
        # self.edges = self.edges.iloc[:int(self.edges.shape[0] * 0.9)]

        graph = sg.StellarGraph(nodes=self.nodes, edges=self.edges)

        # Test split
        edge_splitter_test = EdgeSplitter(graph)
        self.graph_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
            p=0.3, method="global", keep_connected=False, seed=2023
        )

        # for i in range(edge_ids_test.shape[0]):
        #     if edge_labels_test[i] == 0:
        #         edge_ids_test = edge_ids_test[:i]
        #         edge_labels_test = edge_labels_test[:i]
        #         break

        indices = []
        for i in range(edge_ids_test.shape[0]):
            if not ((int(edge_ids_test[i][0]) in test_edges['source'].values) and (
                    int(edge_ids_test[i][1]) in test_edges['target'].values)):
                indices.append(i)

        edge_ids_test = np.delete(edge_ids_test, indices, axis=0)
        edge_labels_test = np.delete(edge_labels_test, indices)

        # Train split
        edge_splitter_train = EdgeSplitter(self.graph_test)
        self.graph_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
            p=0.3, method="global", keep_connected=False, seed=2023
        )

        indices = []
        for i in range(edge_ids_train.shape[0]):
            if ((int(edge_ids_train[i][0]) in test_edges['source'].values) and (
                    int(edge_ids_train[i][1]) in test_edges['target'].values)):
                indices.append(i)

        self.edge_ids_train = np.delete(edge_ids_train, indices, axis=0)
        self.edge_labels_train = np.delete(edge_labels_train, indices)

        # Define an edge splitter on the reduced graph G_test:
        edge_splitter_val = EdgeSplitter(self.graph_test)

        # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G_test, and obtain the
        # reduced graph G_train with the sampled links removed:
        self.graph_val, edge_ids_val, edge_labels_val = edge_splitter_val.train_test_split(
            p=0.3, method="global", keep_connected=False, seed=2023
        )

        # Train iterators
        self.train_gen = GraphSAGELinkGenerator(self.graph_train, batch_size, num_samples, weighted=False, seed=42)
        self.train_flow = self.train_gen.flow(self.edge_ids_train, self.edge_labels_train, shuffle=True)

        self.val_flow = GraphSAGELinkGenerator(self.graph_val, batch_size, num_samples).flow(
            edge_ids_val, edge_labels_val
        )

        # Test iterators
        test_gen = GraphSAGELinkGenerator(self.graph_test, batch_size, num_samples, weighted=False, seed=42)
        self.test_flow = test_gen.flow(edge_ids_test, edge_labels_test, shuffle=True)

        # Model defining - Keras functional API + Stellargraph layers
        graphsage = GraphSAGE(
            layer_sizes=layer_sizes, generator=self.train_gen, bias=bias, dropout=dropout
        )

        x_inp, x_out = graphsage.in_out_tensors()

        prediction = link_classification(
            output_dim=1, output_act="relu", edge_embedding_method="ip"
        )(x_out)

        base_model = keras.Model(inputs=x_inp, outputs=prediction)

        self.model = BaggingEnsemble(
            model=base_model, n_estimators=self.n_estimators, n_predictions=n_predictions
        )

        self.model.compile(
            optimizer=keras.optimizers.legacy.Adam(lr=lr),
            loss=keras.losses.binary_crossentropy,
            metrics=[keras.metrics.BinaryAccuracy(),keras.metrics.Recall(),keras.metrics.AUC(),keras.metrics.Precision()],
        )
        # return number of training and testing examples
        return self.edge_ids_train.shape[0], edge_ids_test.shape[0]

    def set_weights(self,weights):
        for i in range(self.n_estimators):
            self.model.models[i].set_weights(weights[i])

    def get_weights(self):
        weights = []
        for i in range(self.n_estimators):
            weights.append(self.model.models[i].get_weights())
        return weights



    def fit(self, epochs = 20):
        history = self.model.fit(generator=self.train_gen, train_data=self.edge_ids_train,
                                 train_targets=self.edge_labels_train, epochs=epochs, validation_data=self.val_flow,
                                 verbose=1, use_early_stopping=True,
                                 early_stopping_monitor="val_acc")  # Enable early stopping
        return self.get_weights(),history
    
    def evaluate(self):
        train_metrics_mean, train_metrics_std = self.model.evaluate(self.train_flow, verbose=1)
        test_metrics_mean, test_metrics_std = self.model.evaluate(self.test_flow, verbose=1)
        return train_metrics_mean, test_metrics_mean

    def evaluate_one_edge(self):
        edges = pd.read_csv('data/1.csv')
        nodes = pd.read_csv('data/2.csv', index_col=0)
        graph = sg.StellarGraph(nodes=nodes, edges=edges)
        # edge_splitter_test = EdgeSplitter(graph)
        # self.graph_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
        #     p=0.999, method="global", keep_connected=False, seed=2023
        # )
        test_gen = GraphSAGELinkGenerator(graph, 1, [10, 10], weighted=False, seed=42)
        edge_ids = np.array([[300, 512]])
        edge_labels = np.array([0])
        test_flow = test_gen.flow(edge_ids, edge_labels)
        y = self.model.predict(test_flow)
        x = 5

if __name__ == "__main__":

    #path_weights = "./weights/weights_cora.npy"
    #path_nodes = "./data/4_nodes_0.csv"
    #path_edges = "./data/4_edges_0.csv"

    arg_names = [
        'path_weights',
        'path_nodes',
        'path_edges',
        'epochs'
    ]

    args = dict(zip(arg_names, sys.argv[1:]))

    nodes = pd.read_csv(args["path_nodes"],index_col=0)
    #nodes = nodes.astype("float32")

    edges = pd.read_csv(args["path_edges"])
    #edges = edges.astype({"source":"uint32","target":"uint32"})

    logging.warning('####################################### New Training Session #######################################')

    model = Model(nodes,edges)
    model.initialize()

    logging.info('Training started!')
    start = timer()

    new_weights,history = model.fit(int(args["epochs"]))

    end = timer()
    logging.info('Training done!')

    elapsed_time = end -start

    # Save weights
    np.save(args['path_weights'],new_weights)

    eval = model.evaluate()

    f1_train = (2 * eval[0][2] * eval[0][4]) / (eval[0][2] + eval[0][4])
    f1_test = (2 * eval[1][2] * eval[1][4]) / (eval[1][2] + eval[1][4])

    logging.info('Training set : accuracy - %s, recall - %s, AUC - %s, F1 - %s, precision - %s',eval[0][1],eval[0][2],eval[0][3],f1_train,eval[0][4])
    logging.info('Testing set : accuracy - %s, recall - %s, AUC - %s, F1 - %s, precision - %s',eval[1][1],eval[1][2],eval[1][3],f1_test,eval[1][4])
    logging.info('Elapsed time : %s seconds', elapsed_time)
