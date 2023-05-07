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

import tensorflow as tf
from tensorflow import keras
import random as python_random


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
        self.graph_test = None

        self.train_flow = None
        self.test_flow = None

    def initialize(self,**hyper_params):

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

        # # Get future edges for testing
        # test_edges = self.edges.iloc[int(self.edges.shape[0] * 0.9):]
        #
        # self.edges = self.edges.iloc[:int(self.edges.shape[0] * 0.9)]
        #
        # test_graph = sg.StellarGraph(nodes=self.nodes, edges=test_edges)
        # edge_splitter_test = EdgeSplitter(test_graph)
        # self.graph_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
        #     p=0.1, method="global", keep_connected=False, seed=2023
        # )
        #
        # graph = sg.StellarGraph(nodes=self.nodes, edges=self.edges)
        #
        # # Test split
        # # edge_splitter_test = EdgeSplitter(graph)
        # # self.graph_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
        # #     p=0.1, method="global", keep_connected=False, seed=2023
        # # )
        #
        # # Train split
        # edge_splitter_train = EdgeSplitter(graph)
        # self.graph_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
        #     p=0.1, method="global", keep_connected=False, seed=2023
        # )

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

        edge_ids_train = np.delete(edge_ids_train, indices, axis=0)
        edge_labels_train = np.delete(edge_labels_train, indices)

        # Train iterators
        train_gen = GraphSAGELinkGenerator(self.graph_train, batch_size, num_samples, weighted=False, seed=42)
        self.train_flow = train_gen.flow(edge_ids_train, edge_labels_train, shuffle=True)

        # Test iterators
        test_gen = GraphSAGELinkGenerator(self.graph_test, batch_size, num_samples, weighted=False, seed=42)
        self.test_flow = test_gen.flow(edge_ids_test, edge_labels_test, shuffle=True)

        # Model defining - Keras functional API + Stellargraph layers
        graphsage = GraphSAGE(
            layer_sizes=layer_sizes, generator=train_gen, bias=bias, dropout=dropout
        )

        x_inp, x_out = graphsage.in_out_tensors()

        prediction = link_classification(
            output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
        )(x_out)

        self.model = keras.Model(inputs=x_inp, outputs=prediction)

        self.model.compile(
            optimizer=keras.optimizers.Adam(lr=lr),
            loss=keras.losses.binary_crossentropy,
            metrics=[keras.metrics.BinaryAccuracy(),keras.metrics.Recall(),keras.metrics.AUC(),keras.metrics.Precision()],
        )

        # return number of training and testing examples
        return edge_ids_train.shape[0],edge_ids_test.shape[0]

    def set_weights(self,weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def fit(self, epochs = 20):
        history = self.model.fit(self.train_flow, epochs=epochs, verbose=1)
        return self.model.get_weights(),history
    
    def evaluate(self):
        train_metrics = self.model.evaluate(self.train_flow)
        test_metrics = self.model.evaluate(self.test_flow)
        return train_metrics, test_metrics

    def evaluate_one_edge(self):
        # edges = pd.read_csv('data/1.csv')
        # nodes = pd.read_csv('data/2.csv', index_col=0)
        # graph = sg.StellarGraph(nodes=nodes, edges=edges)
        # edge_splitter_test = EdgeSplitter(graph)
        # self.graph_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
        #     p=0.999, method="global", keep_connected=False, seed=2023
        # )
        # graph = sg.StellarGraph(nodes=self.nodes, edges=self.edges)
        # train_gen = GraphSAGELinkGenerator(graph, batch_size, num_samples, weighted=False, seed=42)
        # edge_ids = np.array([[982, 2836], [7211, 971]])
        # edge_labels = np.array([1, 1])
        # x = train_gen.flow(edge_ids, edge_labels, shuffle=True)
        # y = self.model.predict(test_flow)
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
