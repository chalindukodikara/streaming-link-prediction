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
# time.strftime('%l:%M%p %z on %b %d, %Y')

np.random.seed(0)
######## Setup logger ################
# create data folder with the dataset name
folder_path_logs = "logs"
if os.path.exists(folder_path_logs):
    pass
else:
    os.makedirs(folder_path_logs)

folder_path_process = folder_path_logs + "/pre_process"
if os.path.exists(folder_path_process):
    pass
else:
    os.makedirs(folder_path_process)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s : [%(levelname)s]  %(message)s',
    handlers=[
        logging.FileHandler('logs/pre_process/{}.log'.format(str(time.strftime('%m %d %H:%M:%S # %l:%M%p on %b %d, %Y')))),
        logging.StreamHandler(sys.stdout)
    ]
)
######## Our parameters ################
parser = argparse.ArgumentParser('Preprocessing')
parser.add_argument('--dataset_name', type=str, default='wikipedia', help='Dataset name')
parser.add_argument('--partition_id', type=int, default=0, help='Partition ID')
parser.add_argument('--partition_size', type=int, default=2, help='Partition size')
parser.add_argument('--training_batch_size', type=int, default=10240, help='Training batch size')
parser.add_argument('--testing_batch_size', type=int, default=1024, help='Testing batch size')


try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

DATASET_NAME = args.dataset_name
PARTITION_ID = args.partition_id
PARTITION_SIZE = args.partition_size
TRAINING_BATCH_SIZE = args.training_batch_size
TESTING_BATCH_SIZE = args.testing_batch_size
######## Our parameters ################

def main(dataset_name, data_edges, data_nodes, total_size, training_batch_size = 65536, testing_batch_size = 1024):

    # create data folder with the dataset name
    folder_path = "data/" + dataset_name + '_' + str(PARTITION_SIZE) + '_' + str(PARTITION_ID)
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)
        logging.info("Folder path \"" + folder_path + "\" exists")
    else:
        os.makedirs(folder_path)

    current_timestamp = 0
    batch_number = 0
    while current_timestamp <= (total_size-training_batch_size):
        if current_timestamp != 0: # filter training batch
            # data_edges_temp = data_edges.loc[data_edges['timestamp'] < (current_timestamp + testing_batch_size)].loc[data_edges['timestamp'] > (current_timestamp + 1)]
            data_edges_temp = data_edges.iloc[current_timestamp:current_timestamp + testing_batch_size]
            current_timestamp += testing_batch_size
            logging.info('Test batch {} created'.format(batch_number))
        else: # filter each test batch
            # data_edges_temp = data_edges.loc[data_edges['timestamp'] < training_batch_size].loc[data_edges['timestamp'] > 0]
            data_edges_temp = data_edges.iloc[0:training_batch_size-1]
            current_timestamp += training_batch_size-1
            logging.info('Training batch created')



        # get node list of each batch considering edge set, all sources and targets are added to the node list
        nodes_list = []
        for j in range(2):
            nodes_list = nodes_list + data_edges_temp[data_edges_temp.columns[j]].tolist()

        # filter unique nodes
        nodes_set = set(nodes_list)

        # get nodes dataframe considering unique nodes
        data_nodes_temp = data_nodes[data_nodes[data_nodes.columns[0]].isin(nodes_set)]

        # delete unwanted variables
        del nodes_list
        del nodes_set
        gc.collect()

        # drop timestamp column
        # data_edges_temp = data_edges_temp.drop(columns=["timestamp"])

        # don't drop weight column if there is
        # data_edges_temp = data_edges_temp.iloc[:, :]

        # check whether there are nodes in edge list where those nodes are not in node list
        # nodes_list = data_nodes_temp[data_nodes_temp.columns[0]].to_list()
        # for j in range(2):
        #     for k in range(data_edges_temp.shape[0]):
        #         if data_edges_temp[data_edges_temp.columns[j]].iloc[k] in nodes_list:
        #             pass
        #         else:
        #             print("True", j, k, data_edges_temp[data_edges_temp.columns[j]].iloc[k])

        # save
        if batch_number == 0:
            data_edges_temp.to_csv("data/" + dataset_name + '_' + str(PARTITION_SIZE) + '_' + str(PARTITION_ID) + "/" + "0_training_batch_edges.csv", index=False)
            data_nodes_temp.to_csv("data/" + dataset_name + '_' + str(PARTITION_SIZE) + '_' + str(PARTITION_ID) + "/" + "0_training_batch_nodes.csv", index=False)
        else:
            data_edges_temp.to_csv("data/" + dataset_name + '_' + str(PARTITION_SIZE) + '_' + str(PARTITION_ID) + "/" + str(batch_number) + "_test_batch_edges.csv", index=False)
            data_nodes_temp.to_csv("data/" + dataset_name + '_' + str(PARTITION_SIZE) + '_' + str(PARTITION_ID) + "/" + str(batch_number) + "_test_batch_nodes.csv", index=False)
        batch_number += 1


if __name__ == "__main__":
    # read edge list
    edge_list = pd.read_csv('data/' + DATASET_NAME + '_edges_' + str(PARTITION_SIZE) + '_' + str(PARTITION_ID) + '.csv')
    if 'Unnamed: 0' in edge_list: edge_list.pop('Unnamed: 0')
    if 'Unnamed: 0.1' in edge_list: edge_list.pop('Unnamed: 0.1')

    # read node list
    node_list = pd.read_csv('data/' + DATASET_NAME + '_nodes_' + str(PARTITION_SIZE) + '_' + str(PARTITION_ID) + '.csv')
    if 'Unnamed: 0' in node_list: node_list.pop('Unnamed: 0')
    if 'Unnamed: 0.1' in node_list: node_list.pop('Unnamed: 0.1')

    TOTAL_SIZE = edge_list.shape[0]

    if 'timestamp' in edge_list.columns:
        edge_list = edge_list.drop(columns=["timestamp"])

    if 'weight' in edge_list.columns:
        for i in range(2, len(edge_list.columns)):
            if edge_list.columns[i] == 'weight':
                weight_index = i
                break
        edge_list = pd.concat([edge_list.iloc[:, :2], edge_list.iloc[:, weight_index]], axis=1)
    else:
        edge_list = edge_list.iloc[:, :2]
    logging.info('_____________________________________________________ %s: Pre-processing started _____________________________________________________', DATASET_NAME)
    main(dataset_name=DATASET_NAME, data_edges=edge_list, data_nodes=node_list, total_size=TOTAL_SIZE, training_batch_size=TRAINING_BATCH_SIZE, testing_batch_size=TESTING_BATCH_SIZE)
    logging.info('_____________ %s: Pre-processing finished, total size %s, training batch size %s, testing batch size %s _____________', DATASET_NAME, TOTAL_SIZE, TRAINING_BATCH_SIZE, TESTING_BATCH_SIZE)
