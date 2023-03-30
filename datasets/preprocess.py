import numpy as np
import pandas as pd
import gc
import os
import datetime
# pd.to_datetime(int(data.iloc[2, 3]), unit='s')
import argparse
import logging
import sys
import time
import shutil
np.random.seed(0)

######## Our parameters ################
parser = argparse.ArgumentParser('Preprocessing')
parser.add_argument('--dataset_name', type=str, default='dblp', help='Dataset name')
parser.add_argument('--partition_size', type=int, default=4, help='Partition size')
parser.add_argument('--partition_algorithm', type=str, default='hash', help='Partition algorithm')

try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

DATASET_NAME = args.dataset_name
PARTITION_SIZE = args.partition_size
PARTITION_ALGORITHM = args.partition_algorithm
######## Our parameters ################

if __name__ == "__main__":
    #########################dblp 1956-2022 filter ################################
    # initial_edges_list = pd.read_csv('dynamic-datasets/' + DATASET_NAME + '/' + DATASET_NAME + '_edges.csv')
    # if 'Unnamed: 0' in initial_edges_list: initial_edges_list.pop('Unnamed: 0')
    # if 'Unnamed: 0.1' in initial_edges_list: initial_edges_list.pop('Unnamed: 0.1')
    #
    # data_edges_temp = initial_edges_list.loc[initial_edges_list['timestamp'] < 2023].loc[initial_edges_list['timestamp'] > 1955]
    # data_edges_temp.to_csv('dynamic-datasets/' + DATASET_NAME + '/' + DATASET_NAME + '_edges_new.csv', index=False)
    #########################dblp 1956-2022 filter ################################

    # read node list
    initial_nodes_list = pd.read_csv('dynamic-datasets/' + DATASET_NAME + '/' + DATASET_NAME + '_nodes.csv')
    if 'Unnamed: 0' in initial_nodes_list: initial_nodes_list.pop('Unnamed: 0')
    if 'Unnamed: 0.1' in initial_nodes_list: initial_nodes_list.pop('Unnamed: 0.1')

    print('######################', DATASET_NAME, ' processing started', '######################')
    for partition in range(PARTITION_SIZE):
        print('__________________', ' partition id:', partition, ' started ', '__________________')
        path = 'dynamic-datasets/' + DATASET_NAME + '/partitioned_' + str(PARTITION_SIZE) + '/' + PARTITION_ALGORITHM + '/' + DATASET_NAME + '_edges_' + str(PARTITION_SIZE) + '_' + str(partition) + '.csv'
        edge_list = pd.read_csv(path)

        nodes_list = []
        for j in range(2):
            nodes_list = nodes_list + edge_list[edge_list.columns[j]].tolist()

        # filter unique nodes
        nodes_set = set(nodes_list)
        data_nodes_temp = initial_nodes_list[initial_nodes_list[initial_nodes_list.columns[0]].isin(nodes_set)]

        data_nodes_temp.to_csv('dynamic-datasets/' + DATASET_NAME + '/partitioned_' + str(PARTITION_SIZE) + '/' + PARTITION_ALGORITHM + '/' + DATASET_NAME + '_nodes_' + str(PARTITION_SIZE) + '_' + str(partition) + '.csv', index=False)

        print('__________________', ' partition id:', partition, ' finished ', '__________________')



