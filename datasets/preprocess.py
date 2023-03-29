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
parser.add_argument('--dataset_name', type=str, default='wikipedia', help='Dataset name')
parser.add_argument('--partition_size', type=int, default=2, help='Partition size')
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
    # read node list
    initial_nodes_list = pd.read_csv('dynamic-datasets/' + DATASET_NAME + '/' + DATASET_NAME + '_nodes.csv')
    if 'Unnamed: 0' in initial_nodes_list: initial_nodes_list.pop('Unnamed: 0')
    if 'Unnamed: 0.1' in initial_nodes_list: initial_nodes_list.pop('Unnamed: 0.1')

    for partition in range(PARTITION_SIZE):
        path = 'dynamic-datasets/' + DATASET_NAME + '/partitioned_' + str(PARTITION_SIZE) + '/' + PARTITION_ALGORITHM + '/' + DATASET_NAME + '_edges_' + str(PARTITION_SIZE) + '_' + str(partition) + '.csv'
        edge_list = pd.read_csv(path)

        nodes_list = []
        for j in range(2):
            nodes_list = nodes_list + edge_list[edge_list.columns[j]].tolist()

        # filter unique nodes
        nodes_set = set(nodes_list)
        data_nodes_temp = initial_nodes_list[initial_nodes_list[initial_nodes_list.columns[0]].isin(nodes_set)]

        data_nodes_temp.to_csv('dynamic-datasets/' + DATASET_NAME + '/partitioned_' + str(PARTITION_SIZE) + '/' + PARTITION_ALGORITHM + '/' + DATASET_NAME + '_nodes_' + str(PARTITION_SIZE) + '_' + str(partition) + '.csv', index=False)




