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
parser.add_argument('--partition_id', type=int, default=1, help='Partition ID')
parser.add_argument('--partition_size', type=int, default=2, help='Partition size')
parser.add_argument('--training_batch_size', type=int, default=10, help='Training batch size: can be days, hours, weeks, years')
parser.add_argument('--testing_batch_size', type=int, default=1, help='Testing batch size: can be days, hours, weeks, years')


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

def create_wikipedia(data_edges, data_nodes, training_batch_size, testing_batch_size, initial_timestamp=0, last_timestamp=2678373): # In days: 1 , 10
    testing_batch_size = testing_batch_size * 3600 * 24  # convert into seconds and then miliseconds
    training_batch_size = training_batch_size * 3600 * 24
    current_timestamp = initial_timestamp
    batch_number = 0
    while current_timestamp <= (last_timestamp - testing_batch_size):
        if current_timestamp != initial_timestamp:  # filter training batch
            data_edges_temp = data_edges.loc[data_edges['timestamp'] <= (current_timestamp + testing_batch_size)].loc[data_edges['timestamp'] > (current_timestamp)]
            current_timestamp += testing_batch_size
            logging.info('Test batch {} created'.format(batch_number))
        else:  # filter each test batch
            data_edges_temp = data_edges.loc[data_edges['timestamp'] < current_timestamp + training_batch_size].loc[data_edges['timestamp'] >= current_timestamp]
            current_timestamp += training_batch_size - 1
            logging.info('Training batch created')

        if data_edges_temp.empty:
            logging.info('Batch %s: is empty', batch_number)

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
        data_edges_temp = data_edges_temp.drop(columns=["timestamp"])

        # save
        if batch_number == 0:
            data_edges_temp.to_csv("data/" + DATASET_NAME + '_' + str(PARTITION_SIZE) + '_' + str(
                PARTITION_ID) + "/" + "0_training_batch_edges.csv", index=False)
            data_nodes_temp.to_csv("data/" + DATASET_NAME + '_' + str(PARTITION_SIZE) + '_' + str(
                PARTITION_ID) + "/" + "0_training_batch_nodes.csv", index=False)
        else:
            data_edges_temp.to_csv(
                "data/" + DATASET_NAME + '_' + str(PARTITION_SIZE) + '_' + str(PARTITION_ID) + "/" + str(
                    batch_number) + "_test_batch_edges.csv", index=False)
            data_nodes_temp.to_csv(
                "data/" + DATASET_NAME + '_' + str(PARTITION_SIZE) + '_' + str(PARTITION_ID) + "/" + str(
                    batch_number) + "_test_batch_nodes.csv", index=False)
        batch_number += 1
def create_youtube(data_edges, data_nodes, training_batch_size, testing_batch_size, initial_timestamp=1165708800, last_timestamp=1176422400): # In days: 1 , 1
    testing_batch_size = testing_batch_size * 3600 * 24  # convert into seconds
    training_batch_size = training_batch_size * 3600 * 24  # convert into seconds
    current_timestamp = initial_timestamp
    batch_number = 0
    while current_timestamp <= (last_timestamp - testing_batch_size):
        if current_timestamp != initial_timestamp:  # filter training batch
            data_edges_temp = data_edges.loc[data_edges['timestamp'] <= (current_timestamp + testing_batch_size)].loc[data_edges['timestamp'] > (current_timestamp)]
            current_timestamp += testing_batch_size
            logging.info('Test batch {} created'.format(batch_number))
        else:  # filter each test batch
            data_edges_temp = data_edges.loc[data_edges['timestamp'] < current_timestamp + training_batch_size].loc[data_edges['timestamp'] >= current_timestamp]
            current_timestamp += training_batch_size - 1
            logging.info('Training batch created')

        if data_edges_temp.empty:
            logging.info('Batch %s: is empty', batch_number)

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
        data_edges_temp = data_edges_temp.drop(columns=["timestamp"])

        # save
        if batch_number == 0:
            data_edges_temp.to_csv("data/" + DATASET_NAME + '_' + str(PARTITION_SIZE) + '_' + str(
                PARTITION_ID) + "/" + "0_training_batch_edges.csv", index=False)
            data_nodes_temp.to_csv("data/" + DATASET_NAME + '_' + str(PARTITION_SIZE) + '_' + str(
                PARTITION_ID) + "/" + "0_training_batch_nodes.csv", index=False)
        else:
            data_edges_temp.to_csv(
                "data/" + DATASET_NAME + '_' + str(PARTITION_SIZE) + '_' + str(PARTITION_ID) + "/" + str(
                    batch_number) + "_test_batch_edges.csv", index=False)
            data_nodes_temp.to_csv(
                "data/" + DATASET_NAME + '_' + str(PARTITION_SIZE) + '_' + str(PARTITION_ID) + "/" + str(
                    batch_number) + "_test_batch_nodes.csv", index=False)
        batch_number += 1


def create_flights(data_edges, data_nodes, training_batch_size, testing_batch_size, initial_timestamp=0, last_timestamp=121): # In days: 2 , 20
    current_timestamp = initial_timestamp
    batch_number = 0
    while current_timestamp <= (last_timestamp - testing_batch_size):
        if current_timestamp != initial_timestamp:  # filter training batch
            data_edges_temp = data_edges.loc[data_edges['timestamp'] <= (current_timestamp + testing_batch_size)].loc[data_edges['timestamp'] > (current_timestamp)]
            current_timestamp += testing_batch_size
            logging.info('Test batch {} created'.format(batch_number))
        else:  # filter each test batch
            data_edges_temp = data_edges.loc[data_edges['timestamp'] < current_timestamp + training_batch_size].loc[data_edges['timestamp'] >= current_timestamp]
            current_timestamp += training_batch_size - 1
            logging.info('Training batch created')

        if data_edges_temp.empty:
            logging.info('Batch %s: is empty', batch_number)

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
        data_edges_temp = data_edges_temp.drop(columns=["timestamp"])

        # save
        if batch_number == 0:
            data_edges_temp.to_csv("data/" + DATASET_NAME + '_' + str(PARTITION_SIZE) + '_' + str(
                PARTITION_ID) + "/" + "0_training_batch_edges.csv", index=False)
            data_nodes_temp.to_csv("data/" + DATASET_NAME + '_' + str(PARTITION_SIZE) + '_' + str(
                PARTITION_ID) + "/" + "0_training_batch_nodes.csv", index=False)
        else:
            data_edges_temp.to_csv(
                "data/" + DATASET_NAME + '_' + str(PARTITION_SIZE) + '_' + str(PARTITION_ID) + "/" + str(
                    batch_number) + "_test_batch_edges.csv", index=False)
            data_nodes_temp.to_csv(
                "data/" + DATASET_NAME + '_' + str(PARTITION_SIZE) + '_' + str(PARTITION_ID) + "/" + str(
                    batch_number) + "_test_batch_nodes.csv", index=False)
        batch_number += 1

def create_facebook(data_edges, data_nodes, training_batch_size, testing_batch_size, initial_timestamp=1157454929, last_timestamp=1232231923): # In hours: testing batch size = 30 days
    testing_batch_size = testing_batch_size * 3600 * 24 # convert into seconds
    current_timestamp = 0
    batch_number = 0
    while current_timestamp <= (last_timestamp - testing_batch_size):
        if current_timestamp != 0:  # filter training batch
            data_edges_temp = data_edges.loc[data_edges['timestamp'] <= (current_timestamp + testing_batch_size)].loc[data_edges['timestamp'] > (current_timestamp)]
            current_timestamp += testing_batch_size
            logging.info('Test batch {} created'.format(batch_number))
        else:  # filter each test batch
            data_edges_temp = data_edges.loc[data_edges['timestamp'] == current_timestamp]
            current_timestamp += initial_timestamp
            logging.info('Training batch created')

        if data_edges_temp.empty:
            logging.info('Batch %s: is empty', batch_number)

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
        data_edges_temp = data_edges_temp.drop(columns=["timestamp"])

        # save
        if batch_number == 0:
            data_edges_temp.to_csv("data/" + DATASET_NAME + '_' + str(PARTITION_SIZE) + '_' + str(
                PARTITION_ID) + "/" + "0_training_batch_edges.csv", index=False)
            data_nodes_temp.to_csv("data/" + DATASET_NAME + '_' + str(PARTITION_SIZE) + '_' + str(
                PARTITION_ID) + "/" + "0_training_batch_nodes.csv", index=False)
        else:
            data_edges_temp.to_csv(
                "data/" + DATASET_NAME + '_' + str(PARTITION_SIZE) + '_' + str(PARTITION_ID) + "/" + str(
                    batch_number) + "_test_batch_edges.csv", index=False)
            data_nodes_temp.to_csv(
                "data/" + DATASET_NAME + '_' + str(PARTITION_SIZE) + '_' + str(PARTITION_ID) + "/" + str(
                    batch_number) + "_test_batch_nodes.csv", index=False)
        batch_number += 1
def create_dblp(data_edges, data_nodes, training_batch_size, testing_batch_size, initial_timestamp=1956, last_timestamp=2022): # In years: training batch size = 30, testing batch size = 2
    current_timestamp = initial_timestamp
    batch_number = 0
    while current_timestamp <= (last_timestamp - testing_batch_size):
        if current_timestamp != initial_timestamp:  # filter training batch
            data_edges_temp = data_edges.loc[data_edges['timestamp'] <= (current_timestamp + testing_batch_size)].loc[data_edges['timestamp'] > (current_timestamp)]
            current_timestamp += testing_batch_size
            logging.info('Test batch {} created'.format(batch_number))
        else:  # filter each test batch
            data_edges_temp = data_edges.loc[data_edges['timestamp'] < current_timestamp + training_batch_size].loc[data_edges['timestamp'] >= current_timestamp]
            current_timestamp += training_batch_size - 1
            logging.info('Training batch created')

        if data_edges_temp.empty:
            logging.info('Batch %s: is empty', batch_number)


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
        data_edges_temp = data_edges_temp.drop(columns=["timestamp"])

        # save
        if batch_number == 0:
            data_edges_temp.to_csv("data/" + DATASET_NAME + '_' + str(PARTITION_SIZE) + '_' + str(
                PARTITION_ID) + "/" + "0_training_batch_edges.csv", index=False)
            data_nodes_temp.to_csv("data/" + DATASET_NAME + '_' + str(PARTITION_SIZE) + '_' + str(
                PARTITION_ID) + "/" + "0_training_batch_nodes.csv", index=False)
        else:
            data_edges_temp.to_csv(
                "data/" + DATASET_NAME + '_' + str(PARTITION_SIZE) + '_' + str(PARTITION_ID) + "/" + str(
                    batch_number) + "_test_batch_edges.csv", index=False)
            data_nodes_temp.to_csv(
                "data/" + DATASET_NAME + '_' + str(PARTITION_SIZE) + '_' + str(PARTITION_ID) + "/" + str(
                    batch_number) + "_test_batch_nodes.csv", index=False)
        batch_number += 1

def main(dataset_name, data_edges, data_nodes, initial_timestamp, last_timestamp, training_batch_size = 10, testing_batch_size = 2):
    # create data folder with the dataset name
    folder_path = "data/" + dataset_name + '_' + str(PARTITION_SIZE) + '_' + str(PARTITION_ID)
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)
        logging.info("Folder path \"" + folder_path + "\" exists")
    else:
        os.makedirs(folder_path)

    if DATASET_NAME == 'dblp':
        create_dblp(data_edges, data_nodes, training_batch_size, testing_batch_size)
    elif DATASET_NAME == 'facebook':
        create_facebook(data_edges, data_nodes, training_batch_size, testing_batch_size)

    elif DATASET_NAME == 'flights':
        create_flights(data_edges, data_nodes, training_batch_size, testing_batch_size)

    elif DATASET_NAME == 'youtube':
        create_youtube(data_edges, data_nodes, training_batch_size, testing_batch_size)

    elif DATASET_NAME == 'wikipedia':
        create_wikipedia(data_edges, data_nodes, training_batch_size, testing_batch_size)



if __name__ == "__main__":
    # read edge list
    edge_list = pd.read_csv('data/' + DATASET_NAME + '_edges_' + str(PARTITION_SIZE) + '_' + str(PARTITION_ID) + '.csv')
    if 'Unnamed: 0' in edge_list: edge_list.pop('Unnamed: 0')
    if 'Unnamed: 0.1' in edge_list: edge_list.pop('Unnamed: 0.1')

    # read node list
    node_list = pd.read_csv('data/' + DATASET_NAME + '_nodes_' + str(PARTITION_SIZE) + '_' + str(PARTITION_ID) + '.csv')
    if 'Unnamed: 0' in node_list: node_list.pop('Unnamed: 0')
    if 'Unnamed: 0.1' in node_list: node_list.pop('Unnamed: 0.1')

    # Is hardcoded, memory expensive to read this back
    initial_timestamp = 0
    last_timestamp = 0

    if 'weight' in edge_list.columns:
        for i in range(3, len(edge_list.columns)):
            if edge_list.columns[i] == 'weight':
                weight_index = i
                break
        edge_list = pd.concat([edge_list.iloc[:, :3], edge_list.iloc[:, weight_index]], axis=1)
    else:
        edge_list = edge_list.iloc[:, :3]

    logging.info('_____________________________________________________ %s: Pre-processing started _____________________________________________________', DATASET_NAME)
    main(dataset_name=DATASET_NAME, data_edges=edge_list, data_nodes=node_list, initial_timestamp=initial_timestamp, last_timestamp=last_timestamp, training_batch_size=TRAINING_BATCH_SIZE, testing_batch_size=TESTING_BATCH_SIZE)
    logging.info('_____________ %s: Pre-processing finished, training batch size %s, testing batch size %s _____________', DATASET_NAME, TRAINING_BATCH_SIZE, TESTING_BATCH_SIZE)
